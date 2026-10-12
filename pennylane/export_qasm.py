"""
export_qasm.py — Write the PennyLane Fourier-mixer circuit as OpenQASM 3
========================================================================
  python export_qasm.py            → circuit.qasm

The circuit is taken from the training qnode (train.make_qnode) and its QFT
is decomposed by PennyLane into h / cp / swap.  The token input stays
symbolic (`input float[64] x0..x3`, the RY angles of AngleEmbedding); the
learnable RZ angles are the trained `q_rot` from checkpoint.npz (written
after every training step, so this works mid-training too).

PennyLane's own qml.to_openqasm only writes OpenQASM 2 with numeric angles,
which would fix one token's input, hence this small QASM 3 emitter.
"""

import argparse, json, os
import numpy as np
import pennylane as qml

from train import N_QUBITS, CKPT_FILE, make_qnode

GATE_SET = {"RY", "RZ", "Hadamard", "ControlledPhaseShift", "SWAP"}


def trained_rotations(ckpt=CKPT_FILE, summary="loss_pennylane.json"):
    """(q_rot, description of where it came from)"""
    if not os.path.exists(ckpt):
        raise SystemExit(f"no {ckpt} — run train.py first (even --epochs 1)")
    step = "?"
    if os.path.exists(summary):
        with open(summary) as f:
            step = json.load(f).get("final_step", "?")
    return np.load(ckpt)["q_rot"], f"{ckpt}, training step {step}"


def angle(value):
    """Write multiples of pi exactly (the QFT phases), anything else as a float."""
    ratio = float(value) / np.pi
    for denom in (1, 2, 4, 8, 16, 32):
        if abs(ratio * denom - round(ratio * denom)) < 1e-12 and round(ratio * denom):
            num = round(ratio * denom)
            return ("" if num == 1 else "-" if num == -1 else f"{num}*") + "pi" + (
                f"/{denom}" if denom > 1 else "")
    return repr(float(value))


def qasm_lines(tape):
    """One QASM 3 statement per decomposed operation."""
    lines = []
    for op in tape.operations:
        q = [f"q[{w}]" for w in op.wires]
        if op.name == "RY":  # AngleEmbedding(rotation="Y") — the token input
            lines.append(f"ry(x{op.wires[0]}) {q[0]};")
        elif op.name == "RZ":
            lines.append(f"rz({angle(op.parameters[0])}) {q[0]};")
        elif op.name == "Hadamard":
            lines.append(f"h {q[0]};")
        elif op.name == "ControlledPhaseShift":
            lines.append(f"cp({angle(op.parameters[0])}) {q[0]}, {q[1]};")
        elif op.name == "SWAP":
            lines.append(f"swap {q[0]}, {q[1]};")
        else:
            raise ValueError(f"unexpected gate after decomposition: {op.name}")
    return lines


def export(out="circuit.qasm"):
    q_rot, source = trained_rotations()
    qnode = make_qnode("default.qubit")
    tape = qml.workflow.construct_tape(qnode)(np.zeros(N_QUBITS), q_rot)
    (tape,), _ = qml.transforms.decompose(tape, gate_set=GATE_SET)

    text = "\n".join([
        "// Quantum Fourier mixer — PennyLane implementation (pennylane/train.py)",
        f"// {N_QUBITS} qubits:  AngleEmbedding RY(x_i) → RZ(w_i) learnable → qml.QFT → measure Z",
        "// inputs x[i] : first D_MODEL/2 features of one (layer-normed) token",
        f"// RZ angles  : {source}",
        "// The model reads the expectation values <Z_i> (exact statevector), not samples.",
        "// PennyLane wire i = q[i]; <Z_i> on q[i] is output feature i.",
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        *[f"input float[64] x{i};" for i in range(N_QUBITS)],
        f"bit[{N_QUBITS}] c;",
        f"qubit[{N_QUBITS}] q;",
        *qasm_lines(tape),
        *[f"c[{i}] = measure q[{i}];" for i in range(N_QUBITS)],
    ])
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"✓ {out}  ({source})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="circuit.qasm")
    export(ap.parse_args().out)
