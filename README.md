# Incentive-Aware AI Regulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
This repository contains the official implementation for the paper **"Incentive-Aware AI Regulation: A Credal Characterisation"**.

We propose a game-theoretic framework for auditing AI models where the regulator accounts for the **strategic behavior** of model providers. We prove that to prevent "gaming" (regulatory arbitrage), the set of non-compliant distributions must form a **Credal Set** (convex and closed). We also introduce practical regulation mechanisms based on the testing by betting framework.

<p align="center">
<img src="meta_data/credal_strategy.png" width="600" alt="Credal Strategy Diagram">
<br>
<em>From Left-to-Right, Figure 1 (a): A Naive Regulator (Red) testing against a non-convex set of bad models can be gamed by a mixture strategy. A Credal Regulator (Blue) robustly rejects the mixture. Figure 1 (b) and (c) show how a practical regulation mechanism on Waterbirds dataset for models that do not rely on suprious correlation implements a perfect market outcome by making non-compliant agents self exclude (red) and the compliant agents recieve a license to operate in the market.</em>
</p>

## 🚀 Key Insights

1.  **Regulation is a Game:** AI model providers can often have incentives to game the regulations in order to gain access to environments.
2.  **Convexity is Robustness:** If the set of forbidden behaviors is not convex, a strategic agent can mix "bad" models to create a "compliant" one.
3.  **Practical Regulations via Testing by Betting:** We use betting scores (likelihood ratios against the worst-case distribution in the Credal Set) to continuously price the "safety" of a model.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/muandet-lab/incentive-aware-ai-regulation.git](https://github.com/muandet-lab/incentive-aware-ai-regulation.git)
cd incentive-aware-ai-regulation

# Install dependencies
```bash
1. numpy
2. scipy
3. matplotlib
4. scienceplots (for publication-quality figures)
5. torch & torchvision (for Waterbirds experiments)
```
## 📊 Experiments

The repository reproduces the three main empirical results from the paper:

### 1. The Geometry of Gaming (Synthetic)
Demonstrates why regulations must be convex. We simulate a "Bad Agent" who mixes three forbidden distributions to fool a Naive Regulator.

* **Script:** `experiments/gaming_the_regulation/simulation.ipynb`
* **Result:** Comparison of wealth accumulation between Naive (Exploding wealth = False Negative) and Credal (Zero wealth = True Negative) regulators.

## 📂 Project Structure
```bash
.
├── experiments/
│   ├── gaming_the_regulation   # Fig 1a: Naive vs Credal gaming
│   ├── waterbirds              # Fig 1b/c: Waterbirds results
│   ├── fairness/               # Fig 1d: Implicit fairness regulation experiment
│   └── testing/                # Fig 2: Incetive aware tests  
├── meta_data/                  # Saved plots and assets
└── README.md
```
## Citation
If you find this code or our theoretical framework useful, please cite our paper:
```bash
@article{YourName2024Incentive,
  title={Incentive-Aware AI Regulations: Credal Characterisation},
  author={Singh, Anurag and Rodemann, Julian and Verma, Rajeev and Chau, Siu Lun and Muandet, Krikamol},
  journal={arXiv preprint},
  year={2026}
}
```
## 🤝 Contributing
We welcome discussions on the connection between Imprecise Probabilities, Mechanism Design, and AI Safety. Please feel free to open an issue or submit a pull request.