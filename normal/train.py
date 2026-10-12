"""
train.py — Classical Transformer baselines on Shakespeare (no quantum)
======================================================================
Same architecture as the quantum version, with the quantum mixer swapped:

  --mixer attn     Causal single-head self-attention (a "normal" LLM block)
  --mixer fourier  Classical analog of the quantum circuit:
                   learnable phase -> DFT matrix -> tanh, on the same
                   D_MODEL//2 features, with the same N_QUBITS parameters.

Requirements
  pip install torch numpy

Usage
  python train.py --mixer attn                          # 1 GPU (or CPU)
  torchrun --nproc_per_node=2 train.py --mixer attn     # 2 GPUs, data parallel
  Ctrl+C saves and exits; run the same command again to resume (--fresh restarts)
  -> checkpoint_<mixer>.npz, vocab.json, state_<mixer>.pt,
     metrics_<mixer>.jsonl, loss_<mixer>.json   (format: ../metrics.py,
     compare them with  python ../compare.py)
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qllm_common as qc
from qllm_common import SEQ_LEN, D_MODEL, N_QUBITS, FF_DIM

MIXER_KEYS = {"attn": ["wq", "wk", "wv", "wo"], "fourier": ["f_phase"]}

# ──────────────────────────────────────────────────────────
# 1. Parameters (same init as the quantum versions: rng 42, N(0, 0.1))
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, mixer, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.1, shape)

    params = {
        "tok_emb": r(vocab_size, D_MODEL),
        "pos_emb": r(SEQ_LEN, D_MODEL),
        "ff_w1": r(D_MODEL, FF_DIM),
        "ff_b1": r(FF_DIM),
        "ff_w2": r(FF_DIM, D_MODEL),
        "ff_b2": r(D_MODEL),
        "ln1_g": np.ones(D_MODEL),
        "ln1_b": np.zeros(D_MODEL),
        "ln2_g": np.ones(D_MODEL),
        "ln2_b": np.zeros(D_MODEL),
        "lm_w": r(D_MODEL, vocab_size),
        "lm_b": r(vocab_size),
    }
    if mixer == "attn":
        for name in ("wq", "wk", "wv", "wo"):
            params[name] = r(D_MODEL, D_MODEL)
    elif mixer == "fourier":
        params["f_phase"] = r(N_QUBITS)
    else:
        raise ValueError(f"unknown mixer: {mixer}")
    return params


# ──────────────────────────────────────────────────────────
# 2. Classical mixers
# ──────────────────────────────────────────────────────────

# Real/imag parts of the N-point DFT matrix (the classical "QFT")
_j, _k = np.meshgrid(np.arange(N_QUBITS), np.arange(N_QUBITS), indexing="ij")
DFT_C = np.cos(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)
DFT_S = np.sin(2 * np.pi * _j * _k / N_QUBITS) / np.sqrt(N_QUBITS)


class AttentionLM(qc.TinyLM):
    MIXER_KEYS = MIXER_KEYS["attn"]

    def mix(self, x):
        """Causal single-head self-attention. x: (B, T, D_MODEL)."""
        p, T = self.p, x.shape[1]
        q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]
        scores = q @ k.transpose(1, 2) / np.sqrt(D_MODEL)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=-1) @ v @ p["wo"]


class FourierLM(qc.TinyLM):
    MIXER_KEYS = MIXER_KEYS["fourier"]

    def __init__(self, init):
        super().__init__(init)
        self.register_buffer("dft_c", torch.tensor(DFT_C, dtype=qc.DTYPE))
        self.register_buffer("dft_s", torch.tensor(DFT_S, dtype=qc.DTYPE))

    def mix(self, x):
        """
        Classical analog of  RY-embed -> RZ(w) -> QFT -> <Z>.
        First half of each token: phase-rotate, DFT, squash with tanh.
        Second half: passthrough (same as the quantum version).
        """
        phase = self.p["f_phase"]
        x_half, r_half = x[..., :N_QUBITS], x[..., N_QUBITS:]
        re = (x_half * torch.cos(phase)) @ self.dft_c + (x_half * torch.sin(phase)) @ self.dft_s
        return torch.cat([torch.tanh(re), r_half], dim=-1)


def build_model(mixer):
    cls = AttentionLM if mixer == "attn" else FourierLM
    return lambda vocab_size, device, local_rank: cls(init_params(vocab_size, mixer))


if __name__ == "__main__":
    ap = qc.base_argparser()
    ap.add_argument("--mixer", choices=["attn", "fourier"], default="attn")
    a = ap.parse_args()
    from inference import generate

    qc.train(a.mixer, a.mixer, build_model(a.mixer), f"checkpoint_{a.mixer}.npz",
             generate, epochs=a.epochs, batch=a.batch, eval_every=a.eval_every,
             fresh=a.fresh)
