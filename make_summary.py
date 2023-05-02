import wandb
import pandas as pd
from loguru import logger

METRICS = ["acc", "loss", "auroc", "f1", "auc"]
METRICS = [f"{split}/{metric}" for split in ["train", "test"]
           for metric in METRICS]


def get_history(run):
    history = run.history()
    history = history[[
        metric for metric in METRICS if metric in history.columns]]
    return history


def min_selector(histories):
    return {f"min({metric})": [history[metric].min()
                               if metric in history.columns else None
                               for history in histories]
            for metric in METRICS}


def compute_stats(values):
    df = pd.DataFrame(values)
    return {
        f"{agg}({metric})": df[metric].agg(agg)
        for metric in df.columns
        for agg in ("mean", "std")
    }


def make_summary_run_for_group(group: str):
    api = wandb.Api()
    runs = api.runs("mil", {"group": group, "job_type": "train"})
    print(list(runs))
    summary_values = {
        metric: [run.summary.get(metric, None) for run in runs]
        for metric in METRICS
    }
    histories = [get_history(run) for run in runs]

    values = {**summary_values, **min_selector(histories)}
    stats = compute_stats(values)

    # Remove previous summary runs
    for run in api.runs("mil", {"group": group, "job_type": "summary"}):
        logger.info(f"Deleting previous summary run {run.name}")
        run.delete(delete_artifacts=True)

    wandb.init(project="mil", group=group,
               job_type="summary", name=f"summary_{group}")
    wandb.summary.update(stats)
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=str)
    args = parser.parse_args()
    make_summary_run_for_group(args.group)
