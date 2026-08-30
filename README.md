# Machine Learning Analysis of CO2/N2 Capture in Zr-MOFs

This repository contains the data-processing, machine-learning, SHAP interpretation, and geometric confinement analyses used for studying CO2 adsorption and CO2/N2 separation in zirconium-based metal-organic frameworks (Zr-MOFs).

The workflow includes three adsorption targets:

- CO2 uptake
- Heat of adsorption (HoA)
- CO2/N2 selectivity

## Repository Structure

CO2-N2-capture-with-MOFs--ML/
├── README.md
├── requirements.txt
├── data/
│   └── confinement/
├── code/
│   ├── preprocessing.py
│   ├── pipeline_uptake.py
│   ├── pipeline_hoa.py
│   └── pipeline_selectivity.py
└── analysis/
    ├── shap/
    │   ├── shap_uptake.py
    │   ├── shap_hoa.py
    │   └── shap_selectivity.py
    └── confinement/
        └── confinement_analysis.py
        Data Source

The datasets used in this repository were derived from the ARC-MOF database (Burner et al., Chem. Mater. 2023),
which provides MOF structures with DFT-derived partial atomic charges and precomputed chemical and geometric descriptors.
https://doi.org/10.1021/acs.chemmater.2c02485
