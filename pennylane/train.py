"""
train.py — Quantum Transformer trained on Shakespeare
======================================================
Architecture
  - Classical token & positional embeddings (vocab → d_model)
  - Quantum Attention Block  : QFT-based mixing via PennyLane (replaces QKV)
  - Classical Feed-Forward   : two linear layers with ReLU
  - Classical LM-head        : d_model → vocab, cross-entropy loss

The quantum circuit implements a *Fourier mixer* (à la FNet) on
d_model//2  qubits using the PennyLane QFT + learnable RZ rotations,
producing a rich spectral mixing of token features without O(N²)
attention.  All B·T tokens of a batch go through the circuit at once
(parameter broadcasting); the classical layers run on the rank's GPU.

Requirements
  pip install torch pennylane            (optional: pennylane-lightning-gpu)

Usage
  python train.py [--epochs N] [--batch B]              # 1 GPU (or CPU)
  torchrun --nproc_per_node=2 train.py                  # 2 GPUs, data parallel
  Ctrl+C saves and exits; run the same command again to resume (--fresh restarts)
  → checkpoint.npz, vocab.json, state_pennylane.pt, metrics_pennylane.jsonl,
    loss_pennylane.json   (format: ../metrics.py; compare with python ../compare.py)
"""

import os, sys
import numpy as np
import torch
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qllm_common as qc
from qllm_common import SEQ_LEN, D_MODEL, N_QUBITS, FF_DIM

CKPT_FILE = "checkpoint.npz"
MIXER_KEYS = ["q_rot"]

# ──────────────────────────────────────────────────────────
# 1. Quantum Fourier Mixer (PennyLane)
# ──────────────────────────────────────────────────────────


def make_qnode(qdevice):
    dev = qml.device(qdevice, wires=N_QUBITS)
    diff = "backprop" if qdevice == "default.qubit" else "adjoint"

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def quantum_fourier_mixer(x_vec, rot_weights):
        """
        x_vec      : (M, N_QUBITS) float — one row per token (broadcast)
        rot_weights: (N_QUBITS,)   float — learnable per-qubit RZ angles

        Circuit:
          1. AngleEmbedding : maps x_vec → RY rotations on each qubit
          2. Learnable RZ   : parametrised single-qubit phase gates
          3. QFT            : full quantum Fourier transform
          4. Measure <Z>    : returns N_QUBITS expectation values, each (M,)
        """
        qml.AngleEmbedding(x_vec, wires=range(N_QUBITS), rotation="Y")
        for i in range(N_QUBITS):
            qml.RZ(rot_weights[i], wires=i)
        qml.QFT(wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return quantum_fourier_mixer


class QuantumLM(qc.TinyLM):
    MIXER_KEYS = MIXER_KEYS

    def __init__(self, init, qdevice):
        super().__init__(init)
        self.qnode = make_qnode(qdevice)

    def mix(self, x):
        """
        First half of each token embedding (N_QUBITS = D_MODEL//2) goes
        through the circuit; the second half passes through unchanged.
        """
        B, T, _ = x.shape
        # 4-qubit statevectors are simulated on CPU (gradients flow through .to());
        # the classical layers and the DDP gradient all-reduce stay on the GPU.
        x_half = x[..., :N_QUBITS].reshape(B * T, N_QUBITS).cpu()
        self.circuit_evals += B * T
        q_out = torch.stack(self.qnode(x_half, self.p["q_rot"].cpu()), dim=-1)
        q_out = q_out.to(x.device, x.dtype).reshape(B, T, N_QUBITS)
        return torch.cat([q_out, x[..., N_QUBITS:]], dim=-1)


# ──────────────────────────────────────────────────────────
# 2. Model parameters (rng 42, N(0, 0.1) — same as before)
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.1, shape)

    return {
        # Token embedding table: vocab_size × D_MODEL
        "tok_emb": r(vocab_size, D_MODEL),
        # Positional embedding: SEQ_LEN × D_MODEL
        "pos_emb": r(SEQ_LEN, D_MODEL),
        # Quantum mixer rotations: one RZ angle per qubit
        "q_rot": r(N_QUBITS),
        # Feed-forward layer 1 weights & bias
        "ff_w1": r(D_MODEL, FF_DIM),
        "ff_b1": r(FF_DIM),
        # Feed-forward layer 2 weights & bias
        "ff_w2": r(FF_DIM, D_MODEL),
        "ff_b2": r(D_MODEL),
        # Layer-norm (scale & shift) — two instances
        "ln1_g": np.ones(D_MODEL),
        "ln1_b": np.zeros(D_MODEL),
        "ln2_g": np.ones(D_MODEL),
        "ln2_b": np.zeros(D_MODEL),
        # LM head
        "lm_w": r(D_MODEL, vocab_size),
        "lm_b": r(vocab_size),
    }


if __name__ == "__main__":
    ap = qc.base_argparser()
    ap.add_argument("--qdevice", default="default.qubit",
                    help="PennyLane device, e.g. lightning.gpu")
    a = ap.parse_args()
    from inference import generate

    qc.train("pennylane", "quantum",
             lambda vocab_size, device, local_rank:
                 QuantumLM(init_params(vocab_size), a.qdevice),
             CKPT_FILE, generate, epochs=a.epochs, batch=a.batch,
             eval_every=a.eval_every, fresh=a.fresh)
