# Integrating Bayesian Inference and Supervised Learning for Predictive Modeling of Coffee Rust Incidence Among Kenyan Smallholder Farmers

This repository contains the analysis code accompanying the research article:

**"Integrating Bayesian Inference and Supervised Learning for Predictive Modeling of Coffee Rust Incidence Among Kenyan Smallholder Farmers."**

The study evaluates the combined use of Bayesian statistical modeling and supervised machine learning techniques to predict coffee leaf rust incidence across major Arabica-growing counties in Kenya.

---

## 📂 Data Availability

The anonymized dataset used in this study is openly available through Zenodo:

**DOI: https://doi.org/10.5281/zenodo.17861841**

Per KALRO agreements, no personal identifiers or precise geographic coordinates are included.

A placeholder file in `data/README_data.md` describes how to access the dataset.

---

## 📁 Repository Structure

.
├── LICENSE
├── README.md
├── CITATION.cff
├── requirements.txt
├── Dockerfile
├── data/
│ └── README_data.md # links to Zenodo dataset
├── src/
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── bayesian_modeling.py
│ ├── ml_models.py
│ └── evaluation.py
├── notebooks/
│ ├── exploratory_analysis.ipynb
│ ├── model_training.ipynb
│ └── posterior_analysis.ipynb
├── tests/
│ └── test_models.py
└── outputs/
└── figures/


---

## 🔧 Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
▶️ Running the Analysis

Example command to run the full analysis pipeline:

python src/ml_models.py --data path/to/data --output outputs/

🧪 Reproducibility & Environment

A Dockerfile is provided for fully reproducible execution:

docker build -t coffee-rust-model .
docker run -it coffee-rust-model

📖 Citation

If you use this code, please cite the associated Zenodo software DOI (generated upon GitHub release).
Citation metadata is included in CITATION.cff.

📜 License

This project is released under the MIT License (see LICENSE).

🤝 Acknowledgments

This work uses data collected by the Coffee Research Institute (CRI) under KALRO with support from World Coffee Research (WCR). We acknowledge smallholder farmers and research officers who contributed field observations.


---
