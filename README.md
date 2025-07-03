# Anderson Localization and Quantum Disorder in 1D and 2D Systems

**Author**: Iker Marginet Ballester  
**Affiliation**: Université de Toulouse — Laboratoire de Physique Théorique de Toulouse  
**About**: Undergraduate in Physics (L3)  
**Status**: Ongoing | Python simulations & formal analyses in development  
**Languages**: 🇫🇷 French (current) | 🇬🇧 English (upcoming)

---

## 📘 Overview

This repository hosts a formal and progressive research project on *Anderson localization* and related quantum phenomena in disordered systems. The core material stems from an advanced undergraduate thesis that seeks to bridge rigorous theoretical modeling with custom Python simulations.

This work is built with the aim of achieving both pedagogical clarity and scientific reliability — in line with the standards of international academic institutions such as **CERN**.

---

## 📂 Contents

- 📄 `Stage_L3.pdf`  
  Main theoretical report introducing the Anderson model in 1D and quasi-2D, Lyapunov exponents, Inverse Participation Ratio (IPR), spectral statistics, transfer matrix formalism, and applications to electric/magnetic field coupling, among others.

- 📁 `simulations/` *(in progress)*  
  Contains:
  - Small, modular Python codes for each concept and result (e.g., Lyapunov calculation, IPR, spectral unfolding)
  - Comprehensive `main.py` files per topic for reproducibility
  - Auto-generated subfolders with output data and graphs categorized by **disorder strength** and **system regime**

- 📁 `green_function/` *(planned)*  
  A second report focused on **Green’s function formalism** applied to **weakly disordered 1D systems**, including analytical expressions, physical interpretation, and dedicated simulations.

- 📁 `formulary/` *(to be separated)*  
  A complete **formulary and resource compendium** (currently included in `Stage_L3.pdf`, to be extracted as a separate LaTeX document) containing:
  - Definitions and identities (Green’s functions, Bessel, Gamma functions)
  - Proof sketches for unfamiliar mathematical results
  - Annotated graphs and derivations for advanced undergraduate readers

- 📁 `notebooks/` *(upcoming)*  
  Interactive Jupyter notebooks for visualization and educational use (e.g., parametric control of localization length, animation of wavefunction dynamics, energy-level evolution under increasing disorder)

---

## 🧠 Topics Covered

- **1D Anderson Model**: Theory, matrix representation, numerical diagonalization
- **Lyapunov Exponents & Localization Lengths**: Calculation, statistical averaging, transfer matrices
- **Inverse Participation Ratio (IPR)**: Quantitative measure of localization in eigenstates
- **Energy Level Statistics**: GOE vs. Poisson vs. Dirac distributions under different regimes
- **Transfer Matrices (1D and Quasi-2D)**: Stability, propagation, ergodicity and scaling
- **Aubry-André Model & Maryland Model**: Duality, quasi-periodicity, cocycle theory
- **Time Evolution & Dynamical Localization**: Time-dependent Schrödinger resolution
- **Quantum Transport**: Landauer formalism, scaling laws, conductance calculations
- **External Fields**: Anderson-Stark effects, Hofstadter butterfly, Peierls substitution
- **Classical Analogies**: Optical, acoustic, and mechanical disorder propagation
- **Mathematical Appendices**: Green’s functions, Gamma and Bessel functions with applications

---

## 🛠 Technologies & Libraries

All code will be developed using:

- `Python 3.10+`
- `NumPy` – Linear algebra & array manipulation  
- `Matplotlib` – Scientific visualization  
- `SciPy` – Sparse diagonalization, Fourier analysis  
- `JupyterLab` – Interactive exploration and plotting  
- *(No heavy external dependencies; reproducibility prioritized)*

---

## ⚠️ Disclaimer

> This repository is designed as both a **research notebook** and a **reproducible simulation environment**.  

- For each **physical concept or mathematical result**, small and modular Python codes will be included.
- For each **complete topic**, a clean and reproducible `main.py` will be available.
- All **generated results** (e.g., localization lengths, eigenstate profiles, spectral statistics) will be saved in **dedicated subfolders** per **disorder regime** or **parameter set**.
- The main document includes a **mathematical formulary**, which will eventually be released as a **separate LaTeX document**.

---

## 🔍 Vision & Academic Objective

This repository represents my personal approach to rigorous and exploratory physics research. It aligns with my long-term goal to contribute to **condensed matter theory** and **quantum transport phenomena**, particularly in disordered or strongly correlated systems.

All material is crafted with clarity, depth, and reproducibility in mind — in view of pursuing future opportunities in **international collaborative environments**, especially research programs at **CERN** and similar institutions.

---

## 📌 Roadmap

| Milestone | Description | Status |
|----------|-------------|--------|
| `Stage_L3.pdf` | Anderson model theory and application | ✅ |
| Python code structure | Modular simulation framework per topic | ⏳ |
| Data output per regime | Graphs, values, IPR, eigenfunctions | ⏳ |
| Green's function report | Document on weak-disorder regimes | 🔜 |
| Mathematical formulary | Comprehensive appendix in LaTeX | 🔜 |
| English README + PDF | Bilingual support for open access | 🔜 |
| Jupyter Notebooks | Dynamic simulations and sliders | 🔜 |

---

## 📫 Contact

For feedback, academic collaboration, or simulation requests:

- **GitHub**: [github.com/IkerMarginet](https://github.com/IkerMarginet) *(placeholder)*
- **Email**: ikergenki@gmail.com



