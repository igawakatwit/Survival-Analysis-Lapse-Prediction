# Survival-Analysis-Lapse-Prediction
This repository contains the full workflow, codebase, visuals, and deliverables for a MSDS capstone predicting lapse timing in a large health‑insurance portfolio. The goal is to understand when policies lapse, what factors drive lapse risk, and how different survival‑analysis and machine‑learning models compare in calibration, discrimination, and interpretability.

The repo is organized so you can easily explore the project from initial EDA through model development, evaluation, and final reporting.

📁 Code (All Python scripts used in the modeling pipeline):
- mainC.py — Full pipeline: EDA → preprocessing → modeling → calibration → SHAP
- edaC.py — Exploratory data analysis functions
- preprocessingC.py — Data cleaning and preprocessing
- featureengineeringC.py — Feature creation and transformations
- baselineC.py — CoxPH model, PH assumption checks
- ML_modelsC.py — Random Survival Forest model and utilities
- neural_modelC.py — Cox‑Time neural survival model
- robustness_testC.py — Multi‑seed robustness testing
- visualsC.py — All visualization utilities (survival curves, fan charts, SHAP, calibration)

📁 Visuals (Key plots generated throughout the project):
- KM survival curve
- Nelson–Aalen cumulative hazard
- Age distributions and lapse‑event comparisons
- Duration and lapse‑timing distributions
- Calibration plots
- CoxPH, RSF, and Cox‑Time fan charts
- Policy‑level survival curve comparisons (e.g., policy 150, policy 200)
- SHAP explanations for all models
- Schoenfeld residual diagnostics

📁 Dataset (Contains all materials needed to understand and reproduce the modeling work):
- Health Insurance Portfolio Sample — first 50,000 rows of the full dataset (condensed from original)
- Variable Descriptions — definitions and explanations for every original feature included in the dataset.
- Spanish Regional Climate Groupings — mapping of Spanish provinces into homogeneous climatological areas used for geographic feature engineering.
- Percentile Reference Table — lookup table used for percentile‑based transformations and normalization of selected variables. 

📁 Report
Final Report (PDF) — Full write‑up of methodology, experiments, and results

📁 Presentation slides
Presentation (PDF) — Summary of project motivation, modeling approach, and key findings
