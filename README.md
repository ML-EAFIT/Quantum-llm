# Quantum-LLM-Qhack ⚛️🤖

[![PennyLane](https://img.shields.io/badge/PennyLane-FF7A59?style=flat&logo=pennylane&logoColor=white)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=flat&logo=Qiskit&logoColor=white)](https://qiskit.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow?style=flat&logo=huggingface)](https://huggingface.co/jero98772/quantum-qpt-models)

An experimental framework bridging Large Language Models (LLMs) with Quantum Machine Learning (QML) circuits. This project implements hybrid quantum-classical variants using **PennyLane** and **Qiskit**, accompanied by a classical computing baseline variant for performance benchmarking. 


---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed. Clone the repository and install the core dependencies:

```bash
git clone https://github.com/ML-EAFIT/Quantum-llm-Qhack/
cd Quantum-llm-Qhack
pip install -r requirements.txt

```

### Key Dependencies

* **PennyLane** / **PennyLane-Qiskit** plugin for quantum gradient tracking and hybrid backpropagation.
* **Qiskit** for circuit compilation and device handling.
* **Hugging Face Transformers** for loading pre-trained weights and interacting with model artifacts.

---

## 🏋️ Pre-Trained Weights & Checkpoints

The pre-trained model checkpoints and training metrics associated with this repository are hosted publicly on Hugging Face:

* **Hugging Face Hub:** [jero98772/quantum-qpt-models](https://huggingface.co/jero98772/quantum-qpt-models)

To pull and load a pre-trained variant programmatically:

```python
from transformers import AutoModel
# Load the pre-trained quantum-qpt model variant from Hugging Face
model = AutoModel.from_pretrained("jero98772/quantum-qpt-models")

```

---

## 📚 References & Research Background

This repository draws inspiration and architectural designs from the following resources:

* [ArXiv: 2504.13202](https://arxiv.org/pdf/2504.13202) — Quantum-enhanced language model architectures.
* [Kipu Quantum Blog](https://kipu-quantum.com/blog/quantum-intelligence) — Perspectives on quantum intelligence and compression.
* [TechRxiv: 177004376](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177004376.69666864/v1) — Hybrid quantum-classical deep learning pipelines.
* [ArXiv: 2512.02619v1](https://arxiv.org/html/2512.02619v1) & [ArXiv: 2411.15631](https://arxiv.org/abs/2411.15631) — Recent advancements in QML scaling.
* [YouTube Overview Video](https://www.youtube.com/watch?v=7OtH1Cdsswg&vl=es) — Walkthrough on quantum programming paradigms.

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome! Feel free to open an issue or submit a pull request.

