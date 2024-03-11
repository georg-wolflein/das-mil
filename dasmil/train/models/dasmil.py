from typing import Any, Mapping, Optional, Tuple, Sequence, Type, Union, Literal
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from omegaconf import ListConfig

from .distance_aware_attention import DistanceAwareMultiheadAttention

MHA = Union[DistanceAwareMultiheadAttention, nn.MultiheadAttention]
ActivationFunction = Union[Literal["relu"], Literal["gelu"]]


class MHAWrapper(nn.Module):
    def __init__(self, mha: MHA):
        super().__init__()
        self.mha = mha
        self.add_tile_position_kwarg = "coords" in mha.forward.__code__.co_varnames

    def forward(self, *args, coords=None, **kwargs):
        if self.add_tile_position_kwarg:
            return self.mha(*args, coords=coords, **kwargs)[0]
        else:
            return self.mha(*args, **kwargs)[0]


def make_activation(activation: Optional[ActivationFunction]) -> nn.Module:
    if activation is None:
        return nn.Identity()
    return {"relu": nn.ReLU, "gelu": nn.GELU}[activation]()


class DasMIL(nn.Module):
    def __init__(
        self,
        targets: ListConfig,
        d_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_heads: int = 8,
        add_zero_attn: bool = False,
        num_layers: int = 2,
        ffn_expansion: float = 2,
        # agg: str = "mean",  # "mean" or "max"
        # layer_norm=True, # TODO: ablate over pre vs post layer norm, or none at all
        mha: Type[MHA] = DistanceAwareMultiheadAttention,
        mha1: Optional[Type[MHA]] = None,  # if None, use mha
        use_ffn: bool = True,
        ffn_activation: ActivationFunction = "gelu",
        encoder_activation: Optional[ActivationFunction] = "relu",
        dropout_encoder: Optional[float] = None,  # None means same as dropout
    ) -> None:
        super().__init__()
        self.targets = targets

        feedforward_dim = int(hidden_dim * ffn_expansion)
        dropout_encoder = dropout_encoder if dropout_encoder is not None else dropout
        mha1 = mha1 or mha

        self.encoder = nn.Sequential(
            nn.Linear(d_features, hidden_dim), nn.Dropout(dropout_encoder), make_activation(encoder_activation)
        )  # TODO: test without ReLU so it's just a projection; also test without dropout

        self.mha_lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.msas = nn.ModuleList(
            [  # First MSA layer
                MHAWrapper(
                    mha1(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        add_zero_attn=add_zero_attn,
                        kdim=hidden_dim,  # hidden_dim // 2,
                        vdim=hidden_dim,
                    )
                ),
                # Subsequent MSA layers
                *[
                    MHAWrapper(
                        mha(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                            add_zero_attn=add_zero_attn,
                            kdim=hidden_dim,
                            vdim=hidden_dim,
                        )
                    )
                    for _ in range(num_layers - 1)
                ],
            ]
        )
        self.ffns = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Linear(hidden_dim, feedforward_dim),
                        nn.Dropout(dropout),
                        make_activation(ffn_activation),
                        nn.Linear(feedforward_dim, hidden_dim),
                    )
                    if use_ffn
                    else nn.Identity()
                )
                for _ in range(num_layers - 1)
            ]
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
            a = msa_ln(x)  # pre layer norm
            a = msa(a, a, a, coords=coords, need_weights=False)
            x = x + a

            a = ffn_ln(x)
            a = ffn(a)
            x = x + a

        # Aggregate slide-level tokens
        slide_tokens = x.mean(dim=-2)  # B, D

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits
