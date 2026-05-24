# 📉 ChurnShield AI – Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

An **end‑to‑end machine learning project** that predicts telecom customer churn, built with **CatBoost, XGBoost, and LightGBM** and deployed as an interactive **Streamlit web application**.  
This repository contains all code, trained models, and documentation – a complete portfolio piece demonstrating real‑world data science from exploration to deployment.

---

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Why This Matters](#-why-this-matters)
- [The Problem](#-the-problem)
- [Solution & Approach](#-solution--approach)
  
---

## 🎯 Project Overview
**ChurnShield AI** is a machine learning system that identifies customers at high risk of churning **before they leave**.  
It combines a soft‑voting ensemble of three gradient‑boosting classifiers trained on real telecom data and is served through a three‑page Streamlit app:

- **📝 Blog & Findings** – Full project narrative, lessons learned, and business context.
- **📊 Analysis** – Interactive data visualizations, feature importances, and model performance.
- **🧪 Test Model** – Live churn prediction with SHAP explanations of the driving factors.

---

## 💡 Why This Matters
Customer churn silently drains revenue from subscription businesses.  
- Acquiring a new customer costs **5–25× more** than retaining an existing one.  
- A telecom with 1 million customers and a 2% monthly churn rate loses **20,000 accounts every month**.  
- Reducing churn by just **1% can add hundreds of thousands of dollars in annual revenue**.

This project transforms a reactive cost centre into a **proactive retention engine** by predicting churn risk with **~75% recall** – meaning 3 out of 4 would‑be leavers can be targeted with timely intervention offers.

---

## ❓ The Problem
Most companies operate reactively: they only notice churn after customers have left, then try expensive win‑back campaigns.  
The goal here is to answer:  
> *“Can we identify customers who are about to churn — before they actually cancel?”*  

Solving this allows marketing and customer success teams to take **targeted, proactive actions** and save revenue.

---

## 🧠 Solution & Approach
1. **Data Exploration** – Analysed churn patterns across contract types, monthly charges, tenure, and demographics.  
2. **Feature Engineering** – Created a “services adoption” score to reduce multicollinearity.  
3. **Imbalance Handling** – Used **SMOTE** inside a cross‑validation pipeline to generate synthetic churn examples without data leakage.  
4. **Model Tuning** – Optimised `F1‑score` (not accuracy) for three classifiers using **Optuna**.  
5. **Ensemble** – Combined CatBoost, XGBoost, and LightGBM in a **soft‑voting ensemble** for robust, stable predictions.  
6. **Deployment** – Built a Streamlit app with live prediction, SHAP explanations, and a detailed project blog.

---
