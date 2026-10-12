"""
compare.py — Compare the quantum (PennyLane / Qiskit) and classical runs
========================================================================
Reads every */loss_*.json written by the train.py scripts (format and
validation: metrics.py) and produces
  - a summary table on stdout (and comparison.md)
  - comparison.png with four panels:
      train/val loss vs step, val loss vs training time,
      mixer gradient norm vs step (barren-plateau check), val accuracy vs step

Usage
  python compare.py                  # after running the train.py scripts
  python compare.py --smooth 20 --out comparison.png
"""

import argparse, glob, json, os
import numpy as np

from metrics import validate_metrics

# Fixed colour per model, so a missing run never repaints the others
COLORS = {
    "pennylane": "#2a78d6",
    "qiskit": "#eb6834",
    "attn": "#1baf7a",
    "fourier": "#eda100",
}
STYLES = {"qiskit": "--"}  # qiskit ≈ pennylane exactly; dashes keep both visible
ORDER = ["pennylane", "qiskit", "attn", "fourier"]


def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "loss_*.json"))):
        with open(path) as f:
            r = json.load(f)
        errors = validate_metrics(r)
        if errors:  # old schema or broken file
            print(f"skip {path}: {', '.join(errors[:3])} — re-run its train.py")
            continue
        runs[r["backend"]] = r
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0])
                       if kv[0] in ORDER else len(ORDER)))


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return np.asarray(y)
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0.0))
    head = c[1:k] / np.arange(1, k)  # running mean while the window fills
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def summary(runs):
    cols = ["model", "steps", "GPUs", "val loss", "bpc", "val acc", "params",
            "mixer params", "train s", "s/step", "tokens/s", "circuit evals",
            "min mixer grad"]
    rows = []
    for name, r in runs.items():
        rows.append([
            name,
            f"{r['final_step']}/{r['hparams']['epochs']}"
            + ("" if r["completed"] else " (stopped)"),
            f"{r['world_size']} {r['device']}",
            f"{r['val_loss']:.4f}",
            f"{r['bpc']:.3f}",
            f"{r['val_acc']:.3f}",
            f"{r['n_params']:,}",
            f"{r['n_mixer_params']:,}",
            f"{r['train_seconds']:.1f}",
            f"{np.mean(r['step_time']):.3f}",
            f"{r['tokens_per_sec']:,.0f}",
            f"{r['circuit_evals']:,}" if r["circuit_evals"] else "—",
            f"{min(r['mixer_grad_norm']):.2e}",
        ])
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    table = [line(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    table += [line(row) for row in rows]
    return "\n".join(table)


def plot(runs, out, k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), facecolor="#fcfcfb")
    (a_loss, a_time), (a_grad, a_acc) = ax

    for name, r in runs.items():
        c, ls = COLORS.get(name, "#6250d6"), STYLES.get(name, "-")
        steps = np.arange(1, len(r["train_loss"]) + 1)

        a_loss.plot(steps, r["train_loss"], color=c, lw=0.8, alpha=0.18)
        a_loss.plot(steps, smooth(r["train_loss"], k), color=c, lw=2, ls=ls,
                    label=name)
        a_loss.plot(r["eval_step"], r["val_loss_curve"], color=c, lw=0, marker="o",
                    ms=6, mec="#fcfcfb", mew=1.5)

        a_time.plot(np.maximum(r["eval_seconds"], 1e-3), r["val_loss_curve"], color=c,
                    lw=2, ls=ls, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=name)

        a_grad.plot(steps, smooth(r["mixer_grad_norm"], k), color=c, lw=2, ls=ls,
                    label=f"{name} ({r['n_mixer_params']} params)")

        a_acc.plot(r["eval_step"], r["val_acc_curve"], color=c, lw=2, ls=ls,
                   marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=name)

    a_loss.set(title="Loss vs step (line = train, dots = validation)",
               xlabel="step", ylabel="cross-entropy (nats)")
    a_time.set(title="Validation loss vs training time", xlabel="training seconds (log)",
               ylabel="val cross-entropy (nats)", xscale="log")
    a_grad.set(title="Mixer gradient norm (≈0 ⇒ barren plateau)", xlabel="step",
               ylabel="‖∇ mixer params‖ (log)", yscale="log")
    a_acc.set(title="Validation next-char accuracy", xlabel="step", ylabel="top-1 accuracy")
    for a in ax.flat:
        a.set_facecolor("#fcfcfb")
        a.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--smooth", type=int, default=10, help="moving-average window")
    a = ap.parse_args()

    runs = load_runs(a.root)
    if not runs:
        raise SystemExit("no */loss_*.json found — run the train.py scripts first")

    table = summary(runs)
    print(table)
    with open("comparison.md", "w") as f:
        f.write(table + "\n\n## Samples (prompt 'To be')\n")
        for name, r in runs.items():
            f.write(f"\n### {name}\n```\n{r['sample']}\n```\n")
    print(f"\n✓ comparison.md")
    print(f"✓ {plot(runs, a.out, a.smooth)}")


if __name__ == "__main__":
    main()
