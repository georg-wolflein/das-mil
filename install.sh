#!/bin/bash

# Workaround to use CUDA 11.3

poetry install
poetry run python -m pip install --upgrade pip
poetry run python -m pip uninstall -y torch torchvision
poetry run python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113