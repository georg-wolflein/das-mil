# Deep Multiple Instance Learning with Distance-Aware Self-Attention (DAS-MIL)

## Installation

If you're using a GPU, run:
```bash
./install_gpu.sh
```

Otherwise, use:
```bash
./install_gpu.sh
```

## Training

```bash
python3 train.py +experiment=mnist_collage +model=distance_aware_self_attention
```

## Experiments

### CAMELYON16

| experiment                                            |  description |
| ----------------------------------------------------- | ------------ |
| [1head](https://wandb.ai/georgw7777/mil/groups/1head) |  1 head      |
