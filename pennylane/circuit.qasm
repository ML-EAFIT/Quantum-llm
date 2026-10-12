// Quantum Fourier mixer — PennyLane implementation (pennylane/train.py)
// 4 qubits:  AngleEmbedding RY(x_i) → RZ(w_i) learnable → qml.QFT → measure Z
// inputs x[i] : first D_MODEL/2 features of one (layer-normed) token
// RZ angles  : checkpoint.npz, training step 3
// The model reads the expectation values <Z_i> (exact statevector), not samples.
// PennyLane wire i = q[i]; <Z_i> on q[i] is output feature i.
OPENQASM 3.0;
include "stdgates.inc";
input float[64] x0;
input float[64] x1;
input float[64] x2;
input float[64] x3;
bit[4] c;
qubit[4] q;
ry(x0) q[0];
ry(x1) q[1];
ry(x2) q[2];
ry(x3) q[3];
rz(0.08780683136177266) q[0];
rz(-0.030814005896536364) q[1];
rz(-0.06340073279644409) q[2];
rz(0.0881785349933688) q[3];
h q[0];
cp(pi/2) q[1], q[0];
cp(pi/4) q[2], q[0];
cp(pi/8) q[3], q[0];
h q[1];
cp(pi/2) q[2], q[1];
cp(pi/4) q[3], q[1];
h q[2];
cp(pi/2) q[3], q[2];
h q[3];
swap q[0], q[3];
swap q[1], q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
