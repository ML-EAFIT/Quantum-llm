"""
export_qasm.py — Write the Qiskit Fourier-mixer circuit as OpenQASM 3
=====================================================================
  python export_qasm.py            → circuit.qasm

The token input stays symbolic (`input float[64] x0..x3`, the RY angles);
the learnable RZ angles are the trained `q_rot` from checkpoint.npz
(written after every training step, so this works mid-training too).
The QFT is decomposed into h / cp / swap so any OpenQASM 3 tool can read it.
"""

import argparse, json, os
import numpy as np
from qiskit import ClassicalRegister, transpile, qasm3
from qiskit.circuit import Parameter

from train import N_QUBITS, CKPT_FILE, build_circuit


def trained_rotations(ckpt=CKPT_FILE, summary="loss_qiskit.json"):
    """(q_rot, description of where it came from)"""
    if not os.path.exists(ckpt):
        raise SystemExit(f"no {ckpt} — run train.py first (even --epochs 1)")
    step = "?"
    if os.path.exists(summary):
        with open(summary) as f:
            step = json.load(f).get("final_step", "?")
    return np.load(ckpt)["q_rot"], f"{ckpt}, training step {step}"


def export(out="circuit.qasm"):
    q_rot, source = trained_rotations()
    circuit = build_circuit()
    theta = sorted(circuit.parameters, key=lambda p: p.index)
    x = [Parameter(f"x{i}") for i in range(N_QUBITS)]
    bound = circuit.assign_parameters(
        {**{theta[i]: x[i] for i in range(N_QUBITS)},
         **{theta[N_QUBITS + i]: float(q_rot[i]) for i in range(N_QUBITS)}})
    flat = transpile(bound, basis_gates=["ry", "rz", "h", "cp", "swap"],
                     optimization_level=0)
    flat.add_register(ClassicalRegister(N_QUBITS, "c"))
    flat.measure(range(N_QUBITS), range(N_QUBITS))

    header = "\n".join([
        "// Quantum Fourier mixer — Qiskit implementation (quiskit/train.py)",
        f"// {N_QUBITS} qubits:  RY(x_i) embedding → RZ(w_i) learnable → QFT → measure Z",
        "// inputs x[i] : first D_MODEL/2 features of one (layer-normed) token",
        f"// RZ angles  : {source}",
        "// The model reads the expectation values <Z_i> (exact statevector), not samples.",
        "// Qiskit qubit 0 = least-significant; the QFT acts on reversed qubits to match",
        "// PennyLane's wire order, so <Z_i> on qubit i equals PennyLane's wire i.",
        "",
    ])
    with open(out, "w") as f:
        f.write(header + qasm3.dumps(flat) + "\n")
    print(f"✓ {out}  ({source})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="circuit.qasm")
    export(ap.parse_args().out)
