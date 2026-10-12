// Quantum Fourier mixer — Qiskit implementation (quiskit/train.py)
// 4 qubits:  RY(x_i) embedding → RZ(w_i) learnable → QFT → measure Z
// inputs x[i] : first D_MODEL/2 features of one (layer-normed) token
// RZ angles  : checkpoint.npz, training step 3
// The model reads the expectation values <Z_i> (exact statevector), not samples.
// Qiskit qubit 0 = least-significant; the QFT acts on reversed qubits to match
// PennyLane's wire order, so <Z_i> on qubit i equals PennyLane's wire i.
OPENQASM 3.0;
include "stdgates.inc";
input float[64] x0;
input float[64] x1;
input float[64] x2;
input float[64] x3;
bit[4] c;
qubit[4] q;
ry(x0) q[0];
rz(0.0878068313617726) q[0];
h q[0];
ry(x1) q[1];
rz(-0.03081400589653636) q[1];
cp(pi/2) q[0], q[1];
h q[1];
ry(x2) q[2];
rz(-0.06340073279644408) q[2];
cp(pi/4) q[0], q[2];
cp(pi/2) q[1], q[2];
h q[2];
ry(x3) q[3];
rz(0.08817853499336882) q[3];
cp(pi/8) q[0], q[3];
cp(pi/4) q[1], q[3];
cp(pi/2) q[2], q[3];
swap q[2], q[1];
h q[3];
swap q[3], q[0];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];

