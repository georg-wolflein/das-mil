import pandas as pd
import torch
from .visualize import label2char, red


def _compute_stats(selected_bags):
    if len(selected_bags) == 0:
        return None
    pos_pred = [(bag, y_pred)
                for (bag, y_pred) in selected_bags if y_pred > .5]
    neg_pred = [(bag, y_pred)
                for (bag, y_pred) in selected_bags if y_pred <= .5]
    correct = [(bag, y_pred)
               for (bag, y_pred) in selected_bags if (y_pred > .5) == (bag.y == 1.)]
    example = selected_bags[0][0]
    return {
        "acc": len(correct) / len(selected_bags),
        "total": len(selected_bags),
        f"{label2char(0)}pred": len(neg_pred),
        f"{label2char(1)}pred": len(pos_pred),
        "example": "   " + " ".join(red(f"{instance_label:d}", show=key_instance) for instance_label, key_instance in zip(example.instance_labels, example.key_instances))
    }


def print_prediction_stats(predictions, target_numbers: tuple):
    if not isinstance(target_numbers, tuple):
        try:
            target_numbers = tuple(target_numbers)
        except TypeError:
            target_numbers = (target_numbers,)
    df = dict()
    all_predictions = predictions
    for y in (None, 0, 1):
        predictions = [(bag, y_pred)
                       for (bag, y_pred) in all_predictions if y is None or bag.y == y]
        lbl = label2char(y)
        df[f"{lbl} bags"] = _compute_stats(predictions)
        for a in target_numbers:
            selected = [(bag, y_pred)
                        for (bag, y_pred) in predictions if a in bag.instance_labels]
            df[f"{lbl} bags with {a}s"] = _compute_stats(selected)
            selected = [(bag, y_pred)
                        for (bag, y_pred) in predictions if a not in bag.instance_labels]
            df[f"{lbl} bags without {a}s"] = _compute_stats(selected)
            for b in target_numbers:
                if a == b:
                    continue
                selected = [(bag, y_pred)
                            for (bag, y_pred) in predictions if a in bag.instance_labels and b in bag.instance_labels]
                df[f"{lbl} bags with {a}s and {b}s"] = _compute_stats(selected)
                selected = [(bag, y_pred)
                            for (bag, y_pred) in predictions if a in bag.instance_labels and b not in bag.instance_labels]
                df[f"{lbl} bags with {a}s and not {b}s"] = _compute_stats(
                    selected)
                selected = [(bag, y_pred)
                            for (bag, y_pred) in predictions if a in bag.instance_labels or b in bag.instance_labels]
                df[f"{lbl} bags with {a}s or {b}s"] = _compute_stats(
                    selected)
        for i in range(1, 10):
            selected = [(bag, y_pred)
                        for (bag, y_pred) in predictions if bag.key_instances.sum() == i]
            df[f"{lbl} bags with {i} key instance(s)"] = _compute_stats(
                selected)

    df = {k: v for k, v in df.items() if v is not None}
    df = pd.DataFrame(df).T
    df["acc"] = df["acc"] * 100
    int_cols = ["acc", "total", f"{label2char(0)}pred", f"{label2char(1)}pred"]
    df[int_cols] = df[int_cols].astype(int)
    df.rename(columns={"acc": "% acc", "example": "   example"}, inplace=True)
    print()
    import tabulate
    tabulate.PRESERVE_WHITESPACE = True
    print(tabulate.tabulate(df, headers="keys", tablefmt="plain"))
    print()
    print_dist_stats(all_predictions, target_numbers, agg="min")
    print()
    print_dist_stats(all_predictions, target_numbers, agg="max")


def print_dist_stats(predictions, target_numbers: tuple, agg: str = "min", print_correct=True):
    if len(target_numbers) != 2:
        return
    all_predictions = predictions
    for y in (0, 1):
        predictions = [(bag, y_pred)
                       for (bag, y_pred) in all_predictions if y is None or bag.y == y]
        false_min_dist = []
        correct_min_dist = []
        for bag, y_pred in predictions:
            if bag.pos is None:
                continue
            a, b = target_numbers
            ai = torch.argwhere(bag.instance_labels ==
                                a).squeeze(-1).cpu().detach().numpy().tolist()
            bi = torch.argwhere(bag.instance_labels ==
                                b).squeeze(-1).cpu().detach().numpy().tolist()
            edges = [(i, j) for i in ai for j in bi]
            if len(edges) == 0:
                continue
            ai, bi = torch.tensor(edges).T
            pos_a = bag.pos[ai]
            pos_b = bag.pos[bi]
            norm = torch.norm((pos_a - pos_b).float(), dim=-1)
            dist = getattr(norm, agg)().item()
            if (y_pred > .5) == (bag.y > .5):
                correct_min_dist.append(dist)
            else:
                false_min_dist.append(dist)

        if print_correct:
            print(
                f"{label2char(y)}bags that were predicted   correctly had the following {agg} dists: [{', '.join([f'{d:.2f}' for d in sorted(correct_min_dist)])}]")
        print(
            f"{label2char(y)}bags that were predicted incorrectly had the following {agg} dists: [{', '.join([f'{d:.2f}' for d in sorted(false_min_dist)])}]")
