#!/bin/bash

python3 -m venv mil_env
source mil_env/bin/activate
pip install --upgrade pip
pip install wheel numpy
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch_geometric
pip install -r requirements.txt

wandb sweep conf/sweeps/$1

wandb agent charlottemagister/mil/$agent