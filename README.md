# 🤖 Fair AI Pipeline using Databricks, AIF360, and Power BI

This project demonstrates a **fairness-aware AI evaluation pipeline** built using **Databricks**, **AIF360**, **Python**, and **Power BI** — designed to audit ML models for bias and visualize disparities across protected attributes.

---

## 🧠 Use Case

In domains like **healthcare** and **employment**, biased predictions can worsen inequality. This pipeline evaluates model fairness, applies mitigation (Reweighing), and visualizes impact through dashboards and LLM prompt scoring.

---

## 🔧 Tech Stack

| Area         | Tools & Technologies                         |
|--------------|-----------------------------------------------|
| Notebook Dev | Databricks, Python, AIF360                    |
| Visuals      | Power BI (.pbix or mock PNG)                 |
| Evaluation   | Fairness metrics: SPD, DI, AOD, EOD           |
| LLM Bias     | Prompt-based testing + demographic scoring   |

---

## 📂 Folder Structure

```bash
fair-ai-pipeline-llm-databricks/
│
├── notebooks/
│   └── fairness_pipeline_aif360.py     # Bias detection & mitigation notebook
│
├── dashboards/
│   └── fairness_dashboard_mockup.png   # Visual summary of before/after metrics
│
└── README.md
````
📊 Metrics & Mitigation Flow
Notebook includes:

🔍 Bias Detection using:

Statistical Parity Difference (SPD)

Disparate Impact (DI)

Equal Opportunity Difference (EOD)

Average Odds Difference (AOD)

🔄 Mitigation using AIF360's Reweighing algorithm

📈 Visual comparison of fairness before/after mitigation

---
💬 LLM Prompt Scoring Logic
Prompt-based bias testing (e.g., income or hiring predictions)

Variants like:

“A Black woman with 10 years of experience...”

“A White man with 2 years of experience...”

Results scored and compared using fairness metrics + Chi-square

---
🖼️ Sample Dashboard

---

📫 Let’s Connect
This project is part of my portfolio showcasing responsible AI practices across real-world datasets.

📍 Based in Auckland, NZ — open to roles in AI Audit, ML Engineering, or Data Ethics Strategy
🔗 Connect on LinkedIn
