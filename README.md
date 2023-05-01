# Deep Multiple Instance Learning with Distance-Aware Self-Attention (DAS-MIL)

## Installation

```bash
python3 -m venv mil_env
source mil_env/bin/activate
pip install --upgrade pip
pip install wheel numpy
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch_geometric
pip install -r requirements.txt
```

Note that the `requirements.txt` file contains all requirements except those related to PyTorch which need to be installed separately (see above).

## Training

```bash
python3 train.py +experiment=mnist_collage +model=distance_aware_self_attention
```
