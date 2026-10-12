"""
metrics.py — The shared metrics format (schema v2) for every train.py
=====================================================================
Every backend (normal/attn, normal/fourier, pennylane, qiskit) writes the
same two files in its own directory, so ../compare.py can compare them.

1. metrics_<backend>.jsonl — one JSON object per optimisation step
   (append-only, so a resumed run simply continues the file)

     step             int    1-based global step
     train_loss       float  mean cross-entropy (nats) over the *global* batch
     step_time        float  wall-clock seconds for the step (all GPUs, synced)
     tokens_per_sec   float  global batch * seq_len / step_time
     grad_norm        float  L2 norm of the full (rank-averaged) gradient
     mixer_grad_norm  float  L2 norm of the mixer-parameter gradient
     mixer_params     list|null  mixer parameters after the step (if ≤ 16 of them)
     circuit_evals    int    cumulative circuits run for training, all ranks (0 = classical)
     gpu_mem_mb       float  peak GPU memory on rank 0 (0 on CPU)
     val_loss         float|null  held-out cross-entropy, only on eval steps
     val_acc          float|null  held-out top-1 next-char accuracy, only on eval steps

2. loss_<backend>.json — run summary, rebuilt from the JSONL (see SUMMARY_TYPES)
"""

import json, os

SCHEMA_VERSION = 2
TRACK_MIXER_PARAMS_MAX = 16  # only log mixer params one-by-one for small mixers

STEP_KEYS = ("step", "train_loss", "step_time", "tokens_per_sec", "grad_norm",
             "mixer_grad_norm", "mixer_params", "circuit_evals", "gpu_mem_mb",
             "val_loss", "val_acc")

SUMMARY_TYPES = {
    "schema_version": int,
    "backend": str,         # name used by compare.py (pennylane, qiskit, attn, fourier)
    "mixer": str,           # quantum | attn | fourier
    "seed": int,
    "hparams": dict,
    "world_size": int,      # number of processes / GPUs
    "device": str,          # cuda | cpu
    "n_params": int,
    "n_mixer_params": int,
    "final_step": int,
    "completed": bool,      # False if stopped early (Ctrl+C); rerun to resume
    # per-step curves
    "train_loss": list,
    "step_time": list,
    "grad_norm": list,
    "mixer_grad_norm": list,
    "mixer_params": list,
    # per-eval curves
    "eval_step": list,
    "val_loss_curve": list,
    "val_acc_curve": list,
    "eval_seconds": list,   # cumulative training seconds at each eval
    # final numbers
    "val_loss": float,
    "val_acc": float,
    "bpc": float,           # val_loss / ln 2
    "seconds": float,       # wall time, including eval/checkpointing
    "train_seconds": float, # sum of step_time
    "tokens_per_sec": float,
    "circuit_evals": int,
    "sample": str,
}


def validate_metrics(summary):
    """Return a list of problems (empty if the summary matches schema v2)."""
    errors = []
    for key, typ in SUMMARY_TYPES.items():
        if key not in summary:
            errors.append(f"missing {key}")
        elif typ is float and isinstance(summary[key], (int, float)):
            continue
        elif not isinstance(summary[key], typ):
            errors.append(f"{key} should be {typ.__name__}")
    if not errors and summary["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {summary['schema_version']} != {SCHEMA_VERSION}")
    return errors


def read_steps(path, max_step=None):
    """Read JSONL rows, keeping only the last row for each step ≤ max_step."""
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # torn last line after a hard kill
                    continue
                if max_step is None or r["step"] <= max_step:
                    rows[r["step"]] = r
    return [rows[s] for s in sorted(rows)]


def truncate_steps(path, max_step):
    """Rewrite the JSONL keeping steps ≤ max_step (used when resuming)."""
    rows = read_steps(path, max_step)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def append_step(path, row):
    assert set(row) == set(STEP_KEYS), set(row) ^ set(STEP_KEYS)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_summary(rows, **info):
    """Summary dict (schema v2) from the per-step rows plus run info."""
    evals = [r for r in rows if r["val_loss"] is not None]
    cum, eval_seconds = 0.0, []
    for r in rows:
        cum += r["step_time"]
        if r["val_loss"] is not None:
            eval_seconds.append(cum)
    train_seconds = sum(r["step_time"] for r in rows)
    val_loss = evals[-1]["val_loss"] if evals else float("nan")
    tokens = info["hparams"]["batch"] * info["hparams"]["seq_len"] * len(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        **info,
        "final_step": rows[-1]["step"] if rows else 0,
        "train_loss": [r["train_loss"] for r in rows],
        "step_time": [r["step_time"] for r in rows],
        "grad_norm": [r["grad_norm"] for r in rows],
        "mixer_grad_norm": [r["mixer_grad_norm"] for r in rows],
        "mixer_params": [r["mixer_params"] for r in rows if r["mixer_params"] is not None],
        "eval_step": [r["step"] for r in evals],
        "val_loss_curve": [r["val_loss"] for r in evals],
        "val_acc_curve": [r["val_acc"] for r in evals],
        "eval_seconds": eval_seconds,
        "val_loss": val_loss,
        "val_acc": evals[-1]["val_acc"] if evals else float("nan"),
        "bpc": val_loss / 0.6931471805599453,
        "train_seconds": train_seconds,
        "tokens_per_sec": tokens / train_seconds if train_seconds else 0.0,
        "circuit_evals": rows[-1]["circuit_evals"] if rows else 0,
    }
    errors = validate_metrics(summary)
    assert not errors, errors
    return summary
