from typing import Any, Mapping, Optional, Tuple, Sequence, Type, Union
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from omegaconf import ListConfig

from .distance_aware_attention import DistanceAwareMultiheadAttention

MHA = Union[DistanceAwareMultiheadAttention, nn.MultiheadAttention]


class MHAWrapper(nn.Module):
    def __init__(self, mha: MHA):
        super().__init__()
        self.mha = mha
        self.add_tile_position_kwarg = (
            "coords" in mha.forward.__code__.co_varnames
        )

    def forward(self, *args, coords=None, **kwargs):
        if self.add_tile_position_kwarg:
            return self.mha(*args, coords=coords, **kwargs)[0]
        else:
            return self.mha(*args, coords=coords, **kwargs)[0]

class DasMIL(nn.Module):
    def __init__(
        self,
        targets: ListConfig,
        d_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: Optional[int] = None,
        # agg: str = "mean",  # "mean" or "max"
        add_zero_attn=False,
        # layer_norm=True, # TODO: ablate over pre vs post layer norm, or none at all
        mha1: Type[MHA] = DistanceAwareMultiheadAttention,
        mhas: Type[MHA] = DistanceAwareMultiheadAttention,
        ffn: bool = True
    ) -> None:
        super().__init__()
        self.targets = targets

        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.Dropout(dropout), nn.ReLU()) # TODO: test without ReLU so it's just a projection; also test without dropout

        self.mha_lns = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_lns = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        
        self.msas = nn.ModuleList(
            [# First MSA layer
                MHAWrapper(
            mha1(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=hidden_dim, #hidden_dim // 2,
                vdim=hidden_dim,
                add_zero_attn=add_zero_attn,
            )
        ),
        # Subsequent MSA layers
        *[
                MHAWrapper(
                    mhas(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        kdim=hidden_dim,
                        vdim=hidden_dim,
                        add_zero_attn=add_zero_attn,
                    )
                )
                for _ in range(num_layers - 1)
            ]]
        )
        self.ffns = (
            nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()) if ffn else nn.Identity() for _ in range(num_layers - 1)]
            )
        )

        self.heads = nn.ModuleDict(
            {
                target.column: nn.Linear(
                    in_features=hidden_dim,
                    out_features=len(target.classes) if target.type == "categorical" else 1,
                )
                for target in targets
            }
        )
        self.targets = targets

    def forward(self, feats, coords, mask, *args, **kwargs):
        # Linear projection of the input features
        embeddings = self.encoder(feats)  # B, N, D

        coords = coords.to(torch.half)  # B, N, 2  

        # Apply the transformer layers
        x = embeddings
        for msa, ffn, msa_ln, ffn_ln in zip(self.msas, self.ffns, self.mha_lns, self.ffn_lns):
            a = msa_ln(x) # pre layer norm
            a = msa(a, a, a, coords=coords)
            x = x + a

            a = ffn_ln(x)
            a = ffn(a)
            x = x + a
        
        # Aggregate slide-level tokens
        slide_tokens = x.mean(dim=-2)  # B, D

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits