import wandb
from pathlib import Path
from omegaconf import OmegaConf


WANDB_USER = "georgw7777"


def start_sweep(sweep_yaml: Path, project="mil") -> str:
    sweep_configuration = OmegaConf.load(sweep_yaml)
    sweep_configuration = OmegaConf.to_container(
        sweep_configuration, resolve=False)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    return f"{WANDB_USER}/{project}/{sweep_id}"


if __name__ == "__main__":
    sweeps_folder = Path("sweeps") / "cpu"
    output_file = Path("sweeps_cpu.txt")
    with output_file.open("w") as f:
        for sweep_yaml in sweeps_folder.glob("*.yaml"):
            sweep_id = start_sweep(sweep_yaml)
            print(f"Started sweep {sweep_id}")
            f.write(f"{sweep_id} # {sweep_yaml.stem}\n")
