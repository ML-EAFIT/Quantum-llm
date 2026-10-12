"""
train.py — Quantum Transformer trained on Shakespeare (Qiskit edition)
=======================================================================
Architecture
  - Classical token & positional embeddings (vocab → d_model)
  - Quantum Attention Block  : QFT-based Fourier mixer built with Qiskit
  - Classical Feed-Forward   : two linear layers with ReLU
  - Classical LM-head        : d_model → vocab, cross-entropy loss

Gradients
  Qiskit has no built-in autodiff, so the quantum layer is wrapped as a
  `torch.autograd.Function` whose backward pass uses the parameter-shift
  rule (exact for RY / RZ gates).  The rest of the model is differentiated
  by torch.

Estimator
  qiskit-aer-gpu installed → AerEstimator on this rank's GPU, otherwise the
  exact StatevectorEstimator on CPU.  Force one with --estimator.

Requirements
  pip install torch qiskit numpy        (qiskit >= 1.3 for QFTGate)
  optional: pip install qiskit-aer-gpu

Usage
  python train.py [--epochs N] [--batch B]              # 1 GPU (or CPU)
  torchrun --nproc_per_node=2 train.py                  # 2 GPUs, data parallel
  Ctrl+C saves and exits; run the same command again to resume (--fresh restarts)
  → checkpoint.npz, vocab.json, state_qiskit.pt, metrics_qiskit.jsonl,
    loss_qiskit.json   (format: ../metrics.py; compare with python ../compare.py)
"""

import os, sys
import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qllm_common as qc
from qllm_common import SEQ_LEN, D_MODEL, N_QUBITS, FF_DIM

CKPT_FILE = "checkpoint.npz"
MIXER_KEYS = ["q_rot"]

# ──────────────────────────────────────────────────────────
# 1. Quantum Fourier Mixer (Qiskit)
# ──────────────────────────────────────────────────────────


def build_circuit():
    """
    Parameters θ[0:N]   → input angles (RY embedding of one token's half-vector)
                θ[N:2N] → learnable RZ angles

    Circuit:
      1. RY(x_i) on each qubit        (PennyLane AngleEmbedding, rotation="Y")
      2. RZ(w_i) on each qubit        (learnable phases)
      3. QFT on all qubits
      4. Measure <Z_i>
    """
    theta = ParameterVector("θ", 2 * N_QUBITS)
    qc_ = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc_.ry(theta[i], i)
    for i in range(N_QUBITS):
        qc_.rz(theta[N_QUBITS + i], i)
    # PennyLane treats wire 0 as the most-significant qubit; Qiskit treats
    # qubit 0 as least-significant.  Reversing the qubit order reproduces
    # PennyLane's QFT exactly.
    qc_.append(QFTGate(N_QUBITS), list(reversed(range(N_QUBITS))))
    return qc_


def _z_observables():
    """<Z_i> for each qubit i (Qiskit Pauli labels are little-endian)."""
    obs = []
    for i in range(N_QUBITS):
        label = "I" * (N_QUBITS - 1 - i) + "Z" + "I" * i
        obs.append(SparsePauliOp(label))
    return obs


def make_estimator(kind, local_rank):
    """('statevector' | 'aer-gpu' | 'auto', GPU index) → (estimator, circuit, name)"""
    circuit = build_circuit()
    if kind in ("auto", "aer-gpu"):
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator

            if "GPU" not in AerSimulator().available_devices():
                raise RuntimeError("qiskit-aer has no GPU support")
            est = AerEstimator(options={
                "default_precision": 0.0,  # exact expectation values
                "backend_options": {"device": "GPU", "target_gpus": [local_rank]}})
            # Aer has no native QFTGate: decompose it once up front
            return est, transpile(circuit, AerSimulator()), f"aer-gpu:{local_rank}"
        except Exception as e:
            if kind == "aer-gpu":
                raise
            print(f"[rank gpu {local_rank}] Aer GPU unavailable ({e}); using StatevectorEstimator")
    return StatevectorEstimator(), circuit, "statevector"


class QiskitMixer:
    """Runs the circuit for many (x, w) rows at once and counts circuit evaluations."""

    def __init__(self, estimator, circuit):
        self.estimator, self.circuit = estimator, circuit
        self.observables = _z_observables()
        self.evals = 0

    def run(self, params_2d):
        """params_2d: (M, 2*N_QUBITS) → expectation values (M, N_QUBITS)."""
        self.evals += len(params_2d)
        pv = params_2d[:, None, :]  # (M, 1, P) broadcasts against observables (N_QUBITS,)
        result = self.estimator.run([(self.circuit, self.observables, pv)]).result()
        return np.asarray(result[0].data.evs, dtype=np.float64)

    def jacobian(self, x, w):
        """d<Z_j>/dθ_p  with shape (N, N_QUBITS, 2*N_QUBITS), via parameter shift."""
        base = _build_params(x, w)
        N, P = base.shape
        shifted = np.empty((P, 2, N, P))
        shifted[:] = base
        for p in range(P):
            shifted[p, 0, :, p] += np.pi / 2
            shifted[p, 1, :, p] -= np.pi / 2
        evs = self.run(shifted.reshape(-1, P)).reshape(P, 2, N, N_QUBITS)
        jac = 0.5 * (evs[:, 0] - evs[:, 1])  # (P, N, N_QUBITS)
        return np.transpose(jac, (1, 2, 0))  # (N, N_QUBITS, P)


def _build_params(x, w):
    return np.concatenate([x, np.broadcast_to(w, x.shape)], axis=1)


class QuantumFourierMixer(torch.autograd.Function):
    """
    x : (N, N_QUBITS) — one row per token (RY input angles)
    w : (N_QUBITS,)   — shared learnable RZ angles
    returns (N, N_QUBITS) of <Z> expectation values.
    """

    @staticmethod
    def forward(ctx, x, w, mixer):
        ctx.save_for_backward(x, w)
        ctx.mixer = mixer
        xn, wn = x.detach().cpu().numpy(), w.detach().cpu().numpy()
        return torch.from_numpy(mixer.run(_build_params(xn, wn))).to(x)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        jac = ctx.mixer.jacobian(x.detach().cpu().numpy(), w.detach().cpu().numpy())
        gn = g.detach().cpu().numpy()
        gx = np.einsum("nj,njp->np", gn, jac[:, :, :N_QUBITS])
        gw = np.einsum("nj,njp->p", gn, jac[:, :, N_QUBITS:])
        return torch.from_numpy(gx).to(x), torch.from_numpy(gw).to(w), None


class QuantumLM(qc.TinyLM):
    MIXER_KEYS = MIXER_KEYS

    def __init__(self, init, estimator, circuit):
        super().__init__(init)
        self.qmixer = QiskitMixer(estimator, circuit)

    @property
    def circuit_evals(self):  # forward + parameter-shift circuits
        return self.qmixer.evals

    @circuit_evals.setter
    def circuit_evals(self, value):  # TinyLM.__init__ sets 0
        pass

    def mix(self, x):
        """
        The first half of each token embedding goes through the quantum mixer;
        the second half passes through unchanged.
        """
        B, T, _ = x.shape
        x_half = x[..., :N_QUBITS].reshape(B * T, N_QUBITS)
        q_out = QuantumFourierMixer.apply(x_half, self.p["q_rot"], self.qmixer)
        return torch.cat([q_out.reshape(B, T, N_QUBITS), x[..., N_QUBITS:]], dim=-1)


# ──────────────────────────────────────────────────────────
# 2. Model parameters (rng 42, N(0, 0.1) — same as before)
# ──────────────────────────────────────────────────────────


def init_params(vocab_size, seed=42):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.1, shape)

    return {
        "tok_emb": r(vocab_size, D_MODEL),
        "pos_emb": r(SEQ_LEN, D_MODEL),
        "q_rot": r(N_QUBITS),
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


def build_model(estimator_kind):
    def build(vocab_size, device, local_rank):
        est, circuit, name = make_estimator(estimator_kind, local_rank)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"estimator: {name}")
        return QuantumLM(init_params(vocab_size), est, circuit)

    return build


if __name__ == "__main__":
    ap = qc.base_argparser()
    ap.add_argument("--estimator", choices=["auto", "statevector", "aer-gpu"],
                    default="auto")
    a = ap.parse_args()
    from inference import generate

    qc.train("qiskit", "quantum", build_model(a.estimator), CKPT_FILE, generate,
             epochs=a.epochs, batch=a.batch, eval_every=a.eval_every, fresh=a.fresh)
