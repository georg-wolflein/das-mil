# Deep Multiple Instance Learning with Distance-Aware Self-Attention (DAS-MIL)

[Paper link](https://arxiv.org/abs/2305.10552)

## Installation

If you're using a GPU, run:

```bash
./install_gpu.sh
```

Otherwise, use:

```bash
./install_cpu.sh
```

## Training

### MNIST-COLLAGE

```bash
python3 train.py +selected_model/mnist_collage=distance_aware_self_attention +experiment=mnist_collage
```

### MNIST-COLLAGE-INV

```bash
python3 train.py +selected_model/mnist_collage=distance_aware_self_attention +experiment=mnist_collage_inverse
```

### CAMELYON16

```bash
python3 train.py +selected_model/camelyon16=distance_aware_self_attention
```

You can use the option `device=0` to use GPU 0. Full list of options is available at `conf/config.yaml`. We use [hydra](https://hydra.cc) for configuration management.

## Ablations

See the YAML files in `conf/selected_model/mnist_collage_ablations`.
Run `./run_trials_for_selected_models_cpu.sh` to run all the ablations (set `selected_model_type="mnist_collage_ablations"` in the shell script).
