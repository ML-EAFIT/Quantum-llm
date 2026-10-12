"""
qllm_common.py — Shared data, model skeleton and training loop
==============================================================
Used by normal/, pennylane/ and quiskit/ train.py.  Each of those only
defines its *mixer*; everything else (data, multi-GPU, checkpoints,
resume, metrics) lives here so the three runs are directly comparable.

Multi-GPU (data parallel, PyTorch DDP)
  torchrun --nproc_per_node=2 train.py      # one process per GPU
  python train.py                           # single process (GPU 0 or CPU)
  Every rank draws the *same* global batch and keeps rows rank::world, and
  DDP averages the gradients, so the result does not depend on the GPU count.

Checkpoints / resume
  After every step rank 0 atomically writes
    state_<backend>.pt      model + Adam + step + batch RNG (for resuming)
    checkpoint*.npz         plain numpy weights (read by inference.py)
    metrics_<backend>.jsonl one row per step (format: ../metrics.py)
  Ctrl+C finishes the current step, saves, and exits.  Running train.py
  again resumes at the next step.  --fresh starts over.
"""

import os, json, time, signal, argparse, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import metrics

# ──────────────────────────────────────────────────────────
# Hyper-parameters (identical for every backend)
# ──────────────────────────────────────────────────────────
SEQ_LEN = 16
D_MODEL = 8
N_QUBITS = D_MODEL // 2
FF_DIM = 32
BATCH = 32
EPOCHS = 300
LR = 0.02
SEED = 0  # fixes batch order so every model sees the same data
EVAL_EVERY = 20  # steps between validation evaluations
VAL_BATCHES, VAL_BATCH = 10, 64  # fixed held-out batches (rng 123)
SAMPLE_PROMPT, SAMPLE_TOKENS = "To be", 200
VOCAB_FILE = "vocab.json"
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
MAX_CHARS = 80_000
DTYPE = torch.float64  # matches the original numpy runs

# ──────────────────────────────────────────────────────────
# 1. Data
# ──────────────────────────────────────────────────────────


def fetch_shakespeare():
    path = "shakespeare.txt"
    if not os.path.exists(path):
        print("Downloading Shakespeare …")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:MAX_CHARS]


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def make_batches(arr, seq_len, batch_size, rng):
    """Yield (X, Y) arrays of shape (batch, seq_len)."""
    n = len(arr) - seq_len - 1
    while True:
        idxs = rng.integers(0, n, batch_size)
        X = np.stack([arr[i : i + seq_len] for i in idxs])
        Y = np.stack([arr[i + 1 : i + seq_len + 1] for i in idxs])
        yield X, Y


# ──────────────────────────────────────────────────────────
# 2. Distributed setup
# ──────────────────────────────────────────────────────────


def setup_dist():
    """Returns (rank, local_rank, world, device). Works with or without torchrun."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, local_rank, world, device


def is_dist():
    return dist.is_available() and dist.is_initialized()


def all_reduce(value, op="sum", device="cpu"):
    if not is_dist():
        return value
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
    return t.item()


def barrier():
    if is_dist():
        dist.barrier()


# ──────────────────────────────────────────────────────────
# 3. Model skeleton: embed → [mixer] → FF → LM head
# ──────────────────────────────────────────────────────────


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return g * (x - mean) / torch.sqrt(var + eps) + b


class TinyLM(nn.Module):
    """
    Parameters are a flat dict with the same names as the numpy checkpoints,
    so state → checkpoint.npz → inference.py needs no renaming.
    Subclasses set MIXER_KEYS and implement mix(x_norm) → (B, T, D_MODEL).
    """

    MIXER_KEYS = ()

    def __init__(self, init_params):
        super().__init__()
        self.p = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(np.asarray(v), dtype=DTYPE))
             for k, v in init_params.items()})
        self.circuit_evals = 0  # incremented by quantum mixers

    def mix(self, x_norm):
        raise NotImplementedError

    def forward(self, token_ids):
        """token_ids (B, T) → (B, T, vocab) log-probabilities"""
        p = self.p
        T = token_ids.shape[1]
        x = p["tok_emb"][token_ids] + p["pos_emb"][:T]
        x = x + self.mix(layer_norm(x, p["ln1_g"], p["ln1_b"]))
        h = F.relu(layer_norm(x, p["ln2_g"], p["ln2_b"]) @ p["ff_w1"] + p["ff_b1"])
        x = x + h @ p["ff_w2"] + p["ff_b2"]
        return F.log_softmax(x @ p["lm_w"] + p["lm_b"], dim=-1)

    def numpy_params(self):
        return {k: v.detach().cpu().numpy() for k, v in self.p.items()}


def nll(log_p, Y):
    return -log_p.gather(-1, Y.unsqueeze(-1)).mean()


@torch.no_grad()
def evaluate(model, val_batches, device):
    """Mean cross-entropy and top-1 next-char accuracy on fixed held-out batches."""
    losses, hits = [], []
    for X, Y in val_batches:
        X, Y = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
        log_p = model(X)
        losses.append(nll(log_p, Y).item())
        hits.append((log_p.argmax(-1) == Y).double().mean().item())
    return float(np.mean(losses)), float(np.mean(hits))


def l2_norm(tensors):
    tensors = [t for t in tensors if t is not None]
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in tensors)).item()) if tensors else 0.0


# ──────────────────────────────────────────────────────────
# 4. Checkpoints
# ──────────────────────────────────────────────────────────


def atomic_save(path, save_fn):
    tmp = f"{path}.tmp{os.path.splitext(path)[1]}"  # keep extension (np.savez appends .npz)
    save_fn(tmp)
    os.replace(tmp, path)


def save_npz(path, model, vocab_size):
    atomic_save(path, lambda p: np.savez(p, **model.numpy_params(),
                                         vocab_size=np.array(vocab_size)))


# ──────────────────────────────────────────────────────────
# 5. Ctrl+C → finish the step, save, exit
# ──────────────────────────────────────────────────────────

_STOP = False


def _request_stop(signum, frame):
    global _STOP
    if _STOP:  # second Ctrl+C: give up immediately (last finished step is saved)
        raise KeyboardInterrupt
    _STOP = True
    print(f"\n[signal {signum}] finishing the current step, then saving …", flush=True)


def install_signal_handlers():
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


# ──────────────────────────────────────────────────────────
# 6. CLI + training loop
# ──────────────────────────────────────────────────────────


def base_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="total steps")
    ap.add_argument("--batch", type=int, default=BATCH, help="global batch (all GPUs)")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--fresh", action="store_true", help="ignore saved state, start over")
    return ap


def train(backend, mixer, build_model, ckpt_file, generate,
          epochs=EPOCHS, batch=BATCH, eval_every=EVAL_EVERY, fresh=False):
    """
    backend     : name in metrics / file names (pennylane, qiskit, attn, fourier)
    mixer       : quantum | attn | fourier
    build_model : (vocab_size, device, local_rank) → TinyLM
    ckpt_file   : numpy checkpoint read by inference.py
    generate    : inference.generate(prompt, np_params, stoi, itos, n_tokens=…)
    """
    rank, local_rank, world, device = setup_dist()
    main = rank == 0
    log = print if main else (lambda *a, **k: None)
    install_signal_handlers()
    assert batch % world == 0, f"--batch {batch} must be divisible by {world} GPUs"

    state_file = f"state_{backend}.pt"
    jsonl_file = f"metrics_{backend}.jsonl"
    summary_file = f"loss_{backend}.json"

    # ── data (identical on every rank) ─────────────────
    text = fetch_shakespeare() if main else None
    barrier()  # rank 0 downloads first
    text = text or fetch_shakespeare()
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    arr = np.array(encode(text, stoi), dtype=np.int32)
    split = int(0.9 * len(arr))  # 90% train / 10% validation
    train_arr, val_arr = arr[:split], arr[split:]
    rng = np.random.default_rng(SEED)
    batcher = make_batches(train_arr, SEQ_LEN, batch, rng)
    val_batcher = make_batches(val_arr, SEQ_LEN, VAL_BATCH, np.random.default_rng(123))
    val_batches = [next(val_batcher) for _ in range(VAL_BATCHES)]
    if main:
        with open(VOCAB_FILE, "w") as f:
            json.dump({"stoi": stoi, "itos": {int(k): v for k, v in itos.items()}}, f)

    # ── model / optimiser ──────────────────────────────
    model = build_model(vocab_size, device, local_rank).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    n_params = int(sum(p.numel() for p in model.parameters()))
    mixer_params = [model.p[k] for k in model.MIXER_KEYS]
    n_mixer_params = int(sum(p.numel() for p in mixer_params))
    ddp = (nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1 else model)

    # ── resume ─────────────────────────────────────────
    start, train_evals, wall_before = 1, 0, 0.0
    if fresh and main:
        for f in (state_file, jsonl_file):
            if os.path.exists(f):
                os.remove(f)
    barrier()
    if os.path.exists(state_file):
        st = torch.load(state_file, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimiser.load_state_dict(st["optimiser"])
        rng.bit_generator.state = st["rng"]
        start = st["step"] + 1
        train_evals, wall_before = st["circuit_evals"], st["wall_seconds"]
        if main:
            metrics.truncate_steps(jsonl_file, st["step"])
        log(f"↻ resumed from {state_file} at step {start}")
    elif main and os.path.exists(jsonl_file):
        os.remove(jsonl_file)

    log(f"\n[{backend}] vocab={vocab_size}  params={n_params:,}  "
        f"world={world}  device={device.type}  batch={batch} ({batch // world}/rank)")
    log(f"{'Step':>6}  {'Loss':>8}  {'Val':>8}  {'Acc':>6}  {'Time/step':>9}")
    log("-" * 46)

    def save_state(step, wall):
        atomic_save(state_file, lambda p: torch.save({
            "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "rng": rng.bit_generator.state, "step": step,
            "circuit_evals": train_evals, "wall_seconds": wall,
            "backend": backend, "world_size": world}, p))
        save_npz(ckpt_file, model, vocab_size)

    t_session = time.time()
    step, stopped = start - 1, False
    for step in range(start, epochs + 1):
        X, Y = next(batcher)  # same global batch on every rank
        X = torch.as_tensor(X[rank::world], device=device)
        Y = torch.as_tensor(Y[rank::world], device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        evals0 = model.circuit_evals
        ddp.train()
        optimiser.zero_grad(set_to_none=True)
        loss = nll(ddp(X), Y)
        loss.backward()  # DDP averages gradients across ranks here
        grad_norm = l2_norm([p.grad for p in model.parameters()])
        mixer_grad_norm = l2_norm([p.grad for p in mixer_params])
        optimiser.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        loss = all_reduce(loss.item(), device=device) / world
        train_evals += int(all_reduce(model.circuit_evals - evals0, device=device))
        step_time = all_reduce(time.time() - t0, "max", device)

        # all ranks agree on stopping at this step
        stopped = all_reduce(float(_STOP), "max", device) > 0

        if main:
            row = {
                "step": step,
                "train_loss": loss,
                "step_time": step_time,
                "tokens_per_sec": batch * SEQ_LEN / step_time,
                "grad_norm": grad_norm,
                "mixer_grad_norm": mixer_grad_norm,
                "mixer_params": (torch.cat([p.detach().flatten() for p in mixer_params])
                                 .cpu().tolist()
                                 if n_mixer_params <= metrics.TRACK_MIXER_PARAMS_MAX
                                 else None),
                "circuit_evals": train_evals,
                "gpu_mem_mb": (torch.cuda.max_memory_allocated(device) / 1e6
                               if device.type == "cuda" else 0.0),
                "val_loss": None,
                "val_acc": None,
            }
            if step % eval_every == 0 or step in (1, epochs) or stopped:
                model.eval()
                row["val_loss"], row["val_acc"] = evaluate(model, val_batches, device)
                log(f"{step:>6}  {loss:>8.4f}  {row['val_loss']:>8.4f}  "
                    f"{row['val_acc']:>6.3f}  {step_time:>8.3f}s")
            metrics.append_step(jsonl_file, row)
            save_state(step, wall_before + time.time() - t_session)
        barrier()
        if stopped:
            break

    # ── summary (also written on Ctrl+C, with completed=False) ──
    if main:
        wall = wall_before + time.time() - t_session
        rows = metrics.read_steps(jsonl_file)
        sample = generate(SAMPLE_PROMPT, model.numpy_params(), stoi, itos,
                          n_tokens=SAMPLE_TOKENS)
        summary = metrics.build_summary(
            rows,
            backend=backend, mixer=mixer, seed=SEED,
            hparams=dict(seq_len=SEQ_LEN, d_model=D_MODEL, ff_dim=FF_DIM, batch=batch,
                         epochs=epochs, lr=LR, eval_every=eval_every),
            world_size=world, device=device.type,
            n_params=n_params, n_mixer_params=n_mixer_params,
            completed=bool(rows) and rows[-1]["step"] >= epochs,
            seconds=wall, sample=sample,
        )
        def write_summary(path):
            with open(path, "w") as f:
                json.dump(summary, f)

        atomic_save(summary_file, write_summary)
        if summary["completed"]:
            log(f"\n✓ finished {epochs} steps  |  val loss {summary['val_loss']:.4f}  "
                f"|  {wall:.1f}s")
        else:
            log(f"\n■ stopped at step {summary['final_step']}/{epochs} — "
                f"run the same command again to resume")
        log(f"✓ {ckpt_file}, {state_file}, {jsonl_file}, {summary_file}")
        log(f"  sample: {sample[:60]!r}…")
    barrier()
    if is_dist():
        dist.destroy_process_group()
