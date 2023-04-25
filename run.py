from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import nbformat
from pathlib import Path
import shutil
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

ep = ExecutePreprocessor()

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str)
parser.add_argument("--rerun", action="store_true")
parser.add_argument("--trials", type=int, default=5)
parser.set_defaults(rerun=False)

EXPERIMENTS_DIR = Path("experiments")
args = parser.parse_args()
experiment_name = args.name
num_trials = args.trials
rerun = args.rerun
experiment_dir = EXPERIMENTS_DIR / experiment_name

if not rerun:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("train.ipynb", experiment_dir)
else:
    for file in experiment_dir.glob("results_*.csv"):
        file.unlink()

for trial in tqdm(range(num_trials), desc="Trials"):
    nb = nbformat.read(str(experiment_dir / "train.ipynb"), as_version=4)
    results_file = experiment_dir / f"results_{trial}.csv"

    nb.cells[0].source += f"\nRESULTS_FILE = '{results_file}'"

    try:
        ep.preprocess(nb)
    except CellExecutionError as e:
        logger.debug("Error executing the notebook, continuing...")

    with open(experiment_dir / f"train_executed.ipynb", mode="w", encoding="utf-8") as f:
        nbformat.write(nb, f)

dfs = []
for results_file in experiment_dir.glob("results_*.csv"):
    df = pd.read_csv(results_file)
    df["trial"] = int(results_file.stem.split("_")[-1])
    dfs.append(df)
df = pd.concat(dfs)

d_last = df.sort_values("epoch", ascending=False).groupby(
    "trial").first().drop(columns="epoch")
d_min = df.groupby("trial").min().drop(columns="epoch")
d_last_10 = df.sort_values("epoch", ascending=False).groupby("trial").head(10)

print(f"Results for experiment {experiment_name}:")
with (experiment_dir / "results.txt").open("w") as f:
    for col in ["error", "test_error", "loss", "test_loss"]:
        line = f"{col:10s}:    {d_last[col].mean():.3f} ± {d_last[col].std():.3f}       (minimum: {d_min[col].mean():.3f} ± {d_min[col].std():.3f})       (last 10: {d_last_10[col].mean():.3f} ± {d_last_10[col].std():.3f})"
        f.write(line + "\n")
        print(line)
    f.write(f"\n{df.trial.unique().size} trials\n")
