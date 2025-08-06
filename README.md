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
| 🖼**Career Alignment Feature** | Aligns all players’ careers to start at Year 0 for trajectory comparison |
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
