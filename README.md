# 🏏 Predicting the Future of Cricket: Visualizing & Forecasting International Run Scoring Trajectories

### Built by: Gokul Ramanan  
Rising Sophomore @ Northeastern University | Data Science & Mathematics Major

---

## Project Overview

This project combines **interactive dashboards**, **machine learning forecasting**, and **real-world domain knowledge** to analyze and predict the cumulative international run-scoring trajectories of the top 85 batters in cricket history. The model also forecasts where the next generation of elite players will end up by **mid-2030**, with highly accurate projections based on their career phase, recent performance, and country schedules.

---

## What Makes This Project Unique?

| Feature | Why It Stands Out |
|-----------|----------------------|
| **Interactive Panel Dashboard** | Filter by format, country, year range, specific players, and more |
| **Custom Weighted Model** | Outperformed XGBoost in forecasting future batting averages |
| **Backtesting Engine** | Validated across 10+ 5-year time windows using MAPE, SMAPE, R², MASE, and PICP |
| **Real-World Schedule Integration** | Used ICC FTP + historical data to forecast match counts |
| **Full Pipeline** | Predicted Matches → Innings → Runs |
| **Predicted Career Endpoints** | Future projections shown as distinct diamond markers |
| **Career Alignment Feature** | Aligns all players’ careers to start at Year 0 for trajectory comparison |
| **Custom Backend API** | Dynamically filters data for dashboards |
| **All-Format Toggle** | Focus on players across Tests, ODIs, and T20Is |

---

## Prediction Accuracy Summary

The model was evaluated using real backtests and future forecast components:

| Model Component                              | MAPE   | R²     | Within 20% | SMAPE |
|---------------------------------------------|--------|--------|-------------|--------|
| Batting Average (Known Matches + Innings)   | 0.215  | 0.866  | 0.676       | 0.182 |
| Match Count Prediction (per Country)        | 0.199  | 0.694  | 0.602       | 0.186 |
| Innings Prediction (per Player)             | 0.029  | 0.990  | 0.995       | 0.029 |

> Full chart and evaluation results included in `model_metrics_comparison.png` and `back_test.py`.

---

## Methodology Summary

### Step 1: Modeling Run Scoring Ability
- Built a **custom weighted average model** using:
  - Recent Form
  - Career Average
  - Trajectory Average (based on where they are in career)
- Found weights that outperformed XGBoost in multiple time splits.

### Step 2: Match Prediction
- Used ICC FTP + historical match data to train an XGBoost model to predict total matches per country until 2030.

### Step 3: Innings Prediction
- Modeled based on match totals and historical innings/match patterns.
- Achieved **R² = 0.990** and **MAPE = 0.029** in backtests.

### Step 4: Final Projection
- `Projected Runs = Predicted Innings × Predicted Batting Average`
- Predictions added as the 2030 datapoint with a special marker on the dashboard.

---

Interactive Dashboard Links (HuggingFace)
ML Career Prediction: https://huggingface.co/spaces/GokulRamanan/gokul-ml-prediction-runs-dashboard
Top 85 Batters Progression: https://huggingface.co/spaces/GokulRamanan/top-85-international-cricket-batters-run-progression

Medium Article (Coming Soon)
Will fully analyze both dashboards, my methodology for my ML prediction model, my overall work, and future work as well. Stay Tuned!

Contact Info
- Email: gokulramanan@northeastern.edu
- LinkedIn: https://www.linkedin.com/in/gokul-ramanan-83494825b/
- Instagram: https://www.instagram.com/gokulramanan_20/
