# Survival-Analysis-Lapse-Prediction
This repository contains the full workflow, codebase, visuals, and deliverables for a MSDS capstone predicting lapse timing in a large health‑insurance portfolio. The goal is to understand when policies lapse, what factors drive lapse risk, and how different survival‑analysis and machine‑learning models compare in calibration, discrimination, and interpretability.

The repo is organized so you can easily explore the project from initial EDA through model development, evaluation, and final reporting.

📁 code
All Python scripts used in the modeling pipeline:
- mainC.py — Full pipeline: EDA → preprocessing → modeling → calibration → SHAP
- edaC.py — Exploratory data analysis functions
- preprocessingC.py — Data cleaning and preprocessing
- featureengineeringC.py — Feature creation and transformations
- baselineC.py — CoxPH model, PH assumption checks
- ML_modelsC.py — Random Survival Forest model and utilities
- neural_modelC.py — Cox‑Time neural survival model
- robustness_testC.py — Multi‑seed robustness testing
- visualsC.py — All visualization utilities (survival curves, fan charts, SHAP, calibration)

📁 visuals
Key plots generated throughout the project, including:
- KM survival curve
- Nelson–Aalen cumulative hazard
- Age distributions and lapse‑event comparisons
- Duration and lapse‑timing distributions
- Calibration plots
- CoxPH, RSF, and Cox‑Time fan charts
- Policy‑level survival curve comparisons (e.g., policy 150, policy 200)
- SHAP explanations for all models
- Schoenfeld residual diagnostics

📁 report
Final Report (PDF) — Full write‑up of methodology, experiments, and results

📁 presentation slides
Presentation (PDF) — Summary of project motivation, modeling approach, and key findings
