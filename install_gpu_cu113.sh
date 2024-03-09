#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install wheel numpy
pip install lightning torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg_lib==0.2.0+pt112cu113 torch_scatter==2.1.0+pt112cu113 torch_sparse==0.6.16+pt112cu113 torch_cluster==1.6.0+pt112cu113 torch_spline_conv==1.2.1+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch_geometric
pip install -r requirements.txt

# Note that the `requirements.txt` file contains all requirements except those related to PyTorch which need to be installed separately (see above).