import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# --- Load Data ---
top85_df = pd.read_csv("Top_85_International_Run_Scorers.csv")
prediction_players = pd.read_csv("prediction_players.csv")
ftp_df = pd.read_csv("FTP Till 2027 Mid - Sheet1.csv")
match_data = pd.read_csv("FTP Till 2027 Mid - Sheet2.csv")
initial_2024 = pd.read_csv("Initial Matches 2024.csv")

# --- Clean and Prepare ---
top85_df["Debut_Year"] = top85_df.groupby(["Name", "Format"])["Year"].transform("min")
top85_df["Career_Year"] = top85_df["Year"] - top85_df["Debut_Year"] + 1

prediction_players["Debut_Year"] = prediction_players.groupby(["Name", "Format"])["Year"].transform("min")
prediction_players["Career_Year"] = prediction_players["Year"] - prediction_players["Debut_Year"] + 1

# --- Historical Trajectory from Top 85 ---
trajectory_list = []
for year in range(1, 21):
    temp = top85_df[(top85_df["Career_Year"] >= year) & (top85_df["Career_Year"] < year + 5)]
    summary = temp.groupby("Format").agg({"Runs": "sum", "Inns": "sum"}).reset_index()
    summary.rename(columns = {"Runs": "Tradjectory_Runs", "Inns": "Tradjectory_Inns"}, inplace= True)
    summary["Career_Year"] = year
    summary["Trajectory_Avg_RPI"] = (summary["Tradjectory_Runs"] / summary["Tradjectory_Inns"]).round(2)
    trajectory_list.append(summary)
trajectory_df = pd.concat(trajectory_list)

# Given my the XGBoost model's predictions for the most optimal weights
format_weights = {
    "test": {"Trajectory_Avg_RPI": 0.374, "Career_Avg": 0.538, "Recent_Form_Avg": 0.088},
    "odi": {"Trajectory_Avg_RPI": 0.504, "Career_Avg": 0.364, "Recent_Form_Avg": 0.132},
    "t20i": {"Trajectory_Avg_RPI": 0.713, "Career_Avg": 0.188, "Recent_Form_Avg": 0.100}
}

def get_weighted_rpi(row):
    if row["Format"] == "test":
        return (
            0.7 * row["Trajectory_Avg_RPI"] +
            0.25 * row["Career_Avg"] +
            0.05 * row["Recent_Form_Avg"]
        )
    elif row["Format"] == "odi":
        return (
            0.45 * row["Trajectory_Avg_RPI"] +
            0.45 * row["Career_Avg"] +
            0.1 * row["Recent_Form_Avg"]
        )
    else:  # T20I
        return (
            0.3 * row["Trajectory_Avg_RPI"] +
            0.7 * row["Career_Avg"] +
            0.0 * row["Recent_Form_Avg"]
        )

# --- Backtesting: 2015–2019 and 2020–2024 ---
def backtest(df, cutoff_year, test_years):
    career_length = df[df["Year"] <= cutoff_year].groupby(["Name", "Format"]).agg({
        "Year": ["min", "max", "count"]
    }).reset_index()
    career_length.columns = ["Name", "Format", "Start_Year", "End_Year", "Years_Played"]

    eligible = career_length[
        (career_length["Start_Year"] >= cutoff_year - 10) &
        (career_length["Start_Year"] <= cutoff_year - 5)
        ][["Name", "Format"]]
    train_stats = df[(df[["Name", "Format"]].apply(tuple, axis=1).isin(eligible.apply(tuple, axis=1)))
    ]
    recent_years = list(range(cutoff_year - 2, cutoff_year + 1))
    recent = train_stats[train_stats["Year"].isin(recent_years)].groupby(["Name", "Format"]).agg({
        "Runs": "sum", "Inns": "sum", "NO": "sum"
    }).reset_index()
    recent["Recent_Form_Avg"] = (recent["Runs"] / (recent["Inns"] - recent["NO"]).clip(lower=1)).round(2)
    career = train_stats[train_stats["Year"] <= cutoff_year].groupby(["Name", "Format"]).agg({
        "Runs": "sum", "Inns": "sum", "NO": "sum"
    }).reset_index()
    career["Career_Avg"] = (career["Runs"] / (career["Inns"] - career["NO"]).clip(lower=1)).round(2)
    latest = train_stats.groupby(["Name", "Format"])["Year"].max().reset_index()
    latest = latest.merge(
        train_stats.groupby(["Name", "Format"])["Debut_Year"].min().reset_index(), on=["Name", "Format"]
    )
    latest["Career_Year"] = latest["Year"] - latest["Debut_Year"] + 1
    latest = latest.merge(trajectory_df, on=["Career_Year", "Format"], how="left")
    merged = latest.merge(recent, on=["Name", "Format"], how="left")
    merged = merged.merge(career[["Name", "Format", "Career_Avg"]], on=["Name", "Format"], how="left")
    merged["Final_Runs_Per_Inning"] = merged.apply(get_weighted_rpi, axis=1)
    actuals = df[
        (df["Year"] >= test_years[0]) & (df["Year"] <= test_years[1])
        & (df[["Name", "Format"]].apply(tuple, axis=1).isin(eligible.apply(tuple, axis=1)))
    ].groupby(["Name", "Format"]).agg({
        "Runs": "sum", "Inns": "sum"
    }).reset_index().rename(columns={"Runs": "Actual_Runs", "Inns": "Actual_Inns"})

    merged = merged.merge(actuals, on=["Name", "Format"], how="left")

    prev_start = test_years[0] - 5
    prev_end = test_years[0] - 1
    prev_5y = df[(df["Year"] >= prev_start) & (df["Year"] <= prev_end)]
    prev_totals = (prev_5y.groupby(["Name", "Format"])["Runs"].sum().reset_index()
                   .rename(columns={"Runs": "Prev_5y_Runs"}))
    merged = merged.merge(prev_totals, on=["Name", "Format"], how="left")

    merged["Prev_5y_Runs"] = merged["Prev_5y_Runs"].fillna(0)
    merged["Predicted_Runs_5y"] = (merged["Final_Runs_Per_Inning"] * merged["Actual_Inns"]).round(1)
    merged["Abs_Error"] = (merged["Predicted_Runs_5y"] - merged["Actual_Runs"]).abs().round(1)
    merged["MAPE"] = (merged["Abs_Error"] / merged["Actual_Runs"]).round(3)
    return merged


def backtest_innings_prediction(df_players, df_matches):
    results = []

    # Precompute matches per country/format/year
    match_lookup = df_matches.set_index(["Country", "Format", "Year"])["Total Matches"].to_dict()

    # Group players
    for (name, fmt), group in df_players.groupby(["Name", "Format"]):
        group = group.sort_values("Year")
        country = group["Country"].iloc[0]
        years = group["Year"].tolist()

        for i in range(len(years) - 10):  # Skip final year
            # Input years
            input_years = years[i:i + 5]
            future_years = years[i + 5:i + 10]

            if len(future_years) < 5:
                continue

            input_df = group[group["Year"].isin(input_years)]
            future_df = group[group["Year"].isin(future_years)]

            # Total innings
            input_inns = input_df["Inns"].sum()
            future_inns = future_df["Inns"].sum()

            # Matches played by country in each phase
            matches_input = sum([match_lookup.get((country, fmt, y), 0) for y in input_years])
            matches_future = sum([match_lookup.get((country, fmt, y), 0) for y in future_years])

            # Handle edge case: avoid divide-by-zero
            inns_per_match = round(input_inns / matches_input, 3) if matches_input > 0 else 0

            results.append({
                "Name": name,
                "Format": fmt,
                "Country": country,
                "Inns_Past_5": input_inns,
                "Matches_Past_5": matches_input,
                "Inns/Match": inns_per_match,
                "Matches_Future_5": matches_future,
                "Inns_Future_5": future_inns,
            })

    df_results = pd.DataFrame(results)

    # Prepare features and target
    X = df_results[["Matches_Past_5", "Inns_Past_5", "Inns/Match", "Matches_Future_5"]]
    y = df_results["Inns_Future_5"]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluate
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    within_20pct = np.mean(np.abs((y - y_pred) / y) <= 0.2) if not y.empty else 0
    smape = np.mean(np.abs(y_pred - y) / ((np.abs(y) + np.abs(y_pred)) / 2))

    print("\n📊 Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Within 20%: {within_20pct:.3f}")
    print(f"SMAPE: {smape:.3f}")

    return df_results.assign(Predicted_Inns_5y=y_pred.round(1))


def evaluate_model(df, label):
    metrics = []
    for fmt in df['Format'].unique():
        subset = df[df['Format'] == fmt]
        y_true = subset['Actual_Runs']
        y_pred = subset['Predicted_Runs_5y']

        nonzero_mask = y_true != 0
        y_true = y_true[nonzero_mask]
        y_pred = y_pred[nonzero_mask]

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) if not y_true.empty else np.nan
        within_20pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) if not y_true.empty else 0

        metrics.append({
            "Format": fmt,
            "MAPE": round(mape, 3) if not np.isnan(mape) else np.nan,
            "MAE": round(mae, 2),
            "R2_Score": round(r2, 3),
            "Within_20%": f"{within_20pct:.3f}",
            "Period": label
        })
    return pd.DataFrame(metrics)

def weighted_mape(df, actual_col="Actual_Runs", pred_col="Predicted_Runs_5y", weight_col="Actual_Inns"):
    """
    Calculates weighted MAPE using Innings as weights to fairly account for sample size.
    """
    df = df.copy()
    df = df[df[actual_col] != 0]  # Avoid division by zero

    df["abs_perc_error"] = (df[actual_col] - df[pred_col]).abs() / df[actual_col]
    df["weighted_error"] = df["abs_perc_error"] * df[weight_col]

    return df["weighted_error"].sum() / df[weight_col].sum()

def compute_run_std_before_cutoff(df_all, cutoff_year):
    # Filter data to only include seasons before or equal to the cutoff year
    pre_cutoff = df_all[df_all["Year"] <= cutoff_year].copy()

    # Group by player and format, then sum runs for each player in each format
    player_runs = (pre_cutoff.groupby(["Name", "Format"])["Actual_Runs"].sum().reset_index())

    # Compute std dev of 5-year totals by format
    return player_runs.groupby("Format")["Actual_Runs"].std().to_dict()

format_constants = {
    "odi": 875,
    "test": 810,
    "t20i": 375
}

def add_prediction_intervals(df, format_constants):
    # Apply prediction intervals
    df["Predicted_Lower"] = df.apply(
        lambda row: row["Predicted_Runs_5y"] - format_constants.get(row["Format"].lower()),
        axis=1,
    )
    df["Predicted_Upper"] = df.apply(
        lambda row: row["Predicted_Runs_5y"] + format_constants.get(row["Format"].lower()),
        axis=1,
    )

    return df

def picp(y_true, lower_bound, upper_bound):
    within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(within_bounds) * 100  # percentage


def smape(y_true, y_pred):
    return (
         np.mean(
            np.abs(y_pred - y_true)
            / ((np.abs(y_true) + np.abs(y_pred)) / 2)
        )
    )


def mase(y_true, y_pred, y_naive):
    """
    Mean Absolute Scaled Error (MASE)
    - y_true: actual values (e.g., Actual_Runs)
    - y_pred: model predictions (e.g., Predicted_Runs_5y)
    - y_naive: naive forecast (e.g., Previous_5y_Runs)
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))

    if mae_naive == 0:
        return np.nan  # Avoid division by zero

    return round(mae_model / mae_naive, 3)

back_tests = {}
for i in range(1999, 2020, 2):
    backtesting = backtest(top85_df, cutoff_year=i, test_years=(i+1, i+6))
    backtesting.to_csv(f"backtest_{i+1}_{i+5}.csv", index = False)

    back_period = pd.read_csv(f"backtest_{i+1}_{i+5}.csv")
    back_period = back_period.dropna().reset_index()
    period = f"{i+1}-{i+5}"
    back_tests[period] = back_period

df_all = pd.concat(list(back_tests.values()), ignore_index=True)
metrics_eval = []

for j in range(1999, 2020, 2):
    period_key = f"{j + 1}-{j + 5}"
    df_period = back_tests[period_key]

    run_std_dict = compute_run_std_before_cutoff(df_all, j)
    df_period = add_prediction_intervals(df_period, format_constants)

    metrics_results = evaluate_model(df_period, period_key)

    w_mape_df = (
        df_period.groupby("Format")
        .apply(lambda x: pd.Series({
            "Weighted_MAPE": round(weighted_mape(x, weight_col="Actual_Inns"), 3)
        }))
        .reset_index()
    )

    smape_df = (df_period.groupby("Format").apply(lambda x: pd.Series({
            "SMAPE": round(smape(x["Actual_Runs"], x["Predicted_Runs_5y"]), 3)
        }))
        .reset_index())

    picp_df = (
        df_period.groupby("Format").apply(lambda x: pd.Series({
            "PICP (%)": round(picp(x["Actual_Runs"], x["Predicted_Lower"], x["Predicted_Upper"]), 1)
        }))
        .reset_index()
    )

    mase_df = (
        df_period.groupby("Format").apply(lambda x: pd.Series({
            "MASE": mase(x["Actual_Runs"], x["Predicted_Runs_5y"], x["Prev_5y_Runs"])
        }))
        .reset_index()
    )

    # Merge into metrics_results
    metrics_results = (metrics_results.merge(w_mape_df, on="Format", how = "left").merge(picp_df, on="Format", how = "left")
                       .merge(smape_df, on="Format", how = "left").merge(mase_df, on = "Format", how= "left"))

    metrics_eval.append(metrics_results)

final_metrics = pd.concat(metrics_eval, ignore_index=True)
final_metrics.to_csv("temporal_cv_2yr_batting_avg_results.csv", index=False)
df_cv = pd.read_csv("temporal_cv_2yr_results.csv")

cv_lst = list(df_cv.columns)
del cv_lst[0]
del cv_lst[4]

cv_mean_dict = {}
cv_mean_dict["Prediction Type Mean"] = "Batting Average"
for k in range(len(cv_lst)):
    cv_mean_dict[cv_lst[k]] = df_cv[cv_lst[k]].mean()
print(cv_mean_dict)

"""
def learn_weights(df):
    # Prepare output dictionary
    results = {}

    # Loop over formats
    for fmt in ["test", "odi", "t20i"]:
        df_fmt = df[df["Format"] == fmt]
        X = df_fmt[["Recent_Form_Avg", "Career_Avg", "Trajectory_Avg_RPI"]]
        y = df_fmt["Actual_Runs"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        feature_weights = dict(zip(X.columns, model.feature_importances_))
        results[fmt] = {
            "MAPE": round(mape, 4),
            "Weights": {k: round(v, 3) for k, v in feature_weights.items()}
        }

    return results
"""

"""
Predicting Matches Algorithm
"""

# Step 2: Drop 2025 as it's incomplete
df = match_data[match_data["Year"] < 2025]

# Step 3: Aggregate total matches per country-format-year
df_grouped = df.groupby(["Country", "Format", "Year"], as_index=False).agg({"Total Matches": "sum"})

# Step 4: Create training rows
records = []
for (country, format_), group in df_grouped.groupby(["Country", "Format"]):
    group = group.sort_values("Year").reset_index(drop=True)
    years = group["Year"].tolist()

    for i in range(len(years) - 6):  # Need 5 years + 3 future
        past_4 = group.loc[i:i+3]
        next_3 = group.loc[i+4:i+6]

        if len(past_4) < 4 or len(next_3) < 3:
            continue

        avg_matches_past4 = past_4["Total Matches"].mean()
        total_matches_next3 = next_3["Total Matches"].sum()
        start_year = group.loc[i, "Year"]

        records.append({
            "Country": country,
            "Format": format_,
            "Start_Year": start_year,
            "Avg_Matches_Past5": avg_matches_past4,
            "Matches_Next3": total_matches_next3
        })

df_model = pd.DataFrame(records)

# Train model
X_train = df_model[["Avg_Matches_Past5"]]
y_train = df_model["Matches_Next3"]

match_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.3, random_state=42)
match_model.fit(X_train, y_train)

y_pred = match_model.predict(X_train)
mae_matches = mean_absolute_error(y_train, y_pred)
mape_matches = mean_absolute_percentage_error(y_train, y_pred)
r2_matches = r2_score(y_train, y_pred)
within_20pct_matches = np.mean(np.abs((y_train - y_pred) / y_train) <= 0.2) if not y_train.empty else 0
smape_matches = np.mean(np.abs(y_pred - y_train) / ((np.abs(y_train) + np.abs(y_pred)) / 2))

print("MAE:", round(mae_matches, 2))
print("MAPE:", round(mape_matches, 3))
print("R²:", round(r2_matches, 3))
print(f"Within 20%: {within_20pct_matches:.3f}")
print(f"SMAPE: {smape_matches:.3f}")

"""
Predicting Matches Actual
"""
match_data_recent = match_data[(match_data["Year"] <= 2025) & (match_data["Year"] >= 2024)].groupby(["Country", "Format"])["Total Matches"].sum().reset_index()
df_predict = ftp_df.merge(match_data_recent, on = ["Country", "Format"], how = "left")
df_predict = df_predict.merge(initial_2024, on = ["Country", "Format"], how = "left")
df_predict["2024-2027 Total"] = df_predict["Estimated Matches"] + df_predict["Total Matches"] - df_predict["Matches"]
print(df_predict)

future_records = []
for i, group in df_predict.iterrows():
    avg_recent_matches = group["2024-2027 Total"] / 3
    predicted_matches = round(match_model.predict([[avg_recent_matches]])[0], 0)

    future_records.append({
        "Country": group["Country"],
        "Format": group["Format"],
        "Avg_2024_27": avg_recent_matches,
        "Predicted_2027_30": predicted_matches
    })

df_predicted_matches = pd.DataFrame(future_records)
df_predict = df_predict.merge(df_predicted_matches, on = ["Country", "Format"], how = "left")
df_predict["Final Predicted Matches"] = df_predict["Estimated Matches"] + df_predict["Predicted_2027_30"]

clean_df = top85_df[(top85_df["Year"] >= 2007) & (top85_df["Career_Year"] != 1)]
cleaner_df = clean_df.groupby(["Name", "Format"]).agg({"Career_Year": "max"}).reset_index().rename(columns = {"Career_Year": "Career_Max"})
cleaner_df_two = clean_df.groupby(["Name", "Format"]).agg({"Career_Year": "min"}).reset_index().rename(columns = {"Career_Year": "Career_Min"})
cleaner_df = cleaner_df.merge(cleaner_df_two, on = ["Name", "Format"], how = "left")
cleaner_df["Career Span (since 2007)"] = cleaner_df["Career_Max"] - cleaner_df["Career_Min"]
clean_backtest_df = cleaner_df[cleaner_df["Career Span (since 2007)"] >= 10]

# Create a key column for filtering
clean_df["Name_Format"] = clean_df["Name"] + "_" + clean_df["Format"]
clean_backtest_df["Name_Format"] = clean_backtest_df["Name"] + "_" + clean_backtest_df["Format"]

# Filter clean_df to only include players in clean_backtest_df
filtered_df = clean_df[clean_df["Name_Format"].isin(clean_backtest_df["Name_Format"])].copy()

# Drop helper column if you want
filtered_df.drop(columns=["Name_Format"], inplace=True)
clean_backtest_df.drop(columns=["Name_Format"], inplace=True)

# Step 1: Train innings prediction model
df_innings_model = backtest_innings_prediction(filtered_df, match_data)

# Step 2: Train the model and save it
X_train = df_innings_model[["Matches_Past_5", "Inns_Past_5", "Inns/Match", "Matches_Future_5"]]
y_train = df_innings_model["Inns_Future_5"]

innings_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
innings_model.fit(X_train, y_train)

def projection(df_players, df_ftp):
    current_year = 2025
    recent_years = list(range(current_year - 4, current_year + 1))
    recent = df_players[df_players["Year"].isin(recent_years)].groupby(["Name", "Format"]).agg({
        "Runs": "sum", "Inns": "sum", "NO": "sum", "Mat": "sum"
    }).reset_index()
    recent.rename(columns = {"Runs": "Recent_runs", "Inns": "Recent_inns", "NO": "Recent_NO", "Mat": "Recent_Mat"}, inplace = True)
    recent["Recent_Form_Avg"] = (recent["Recent_runs"] / (recent["Recent_inns"] - recent["Recent_NO"]).clip(lower=1)).round(2)

    career = df_players.groupby(["Name", "Format"]).agg({"Runs": "sum", "Inns": "sum", "NO": "sum", "Mat": "sum"
    }).reset_index()
    career["Career_Avg"] = (career["Runs"] / (career["Inns"] - career["NO"]).clip(lower=1)).round(2)
    career.rename(columns = {"Runs": "Career_Runs", "Mat": "Career_Matches", "Inns": "Career_Innings"}, inplace = True)
    latest = df_players.groupby(["Name", "Format", "Country"])["Year"].max().reset_index()
    latest = latest.merge(
        df_players.groupby(["Name", "Format", "Country"])["Debut_Year"].min().reset_index(), on=["Name", "Format", "Country"]
    )
    latest["Career_Year"] = latest["Year"] - latest["Debut_Year"] + 1
    latest = latest.merge(trajectory_df, on=["Career_Year", "Format"], how="left")
    merged = latest.merge(recent, on=["Name", "Format"], how="left")
    merged = merged.merge(career[["Name", "Format", "Career_Avg", "Career_Runs", "Career_Matches", "Career_Innings"]],
                          on=["Name", "Format"], how="left")
    merged["Final_Runs_Per_Inning"] = merged.apply(get_weighted_rpi, axis=1)

    # --- Merge future matches ---
    merged = merged.merge(df_ftp[["Country", "Format", "Final Predicted Matches"]], on=["Country", "Format"], how="left")

    # --- Estimate Inns/Match from past 5 years ---
    merged["Inns_Past_5"] = recent["Recent_inns"]
    merged["Matches_Past_5"] = recent["Recent_Mat"]
    merged["Inns/Match"] = merged.apply(
    lambda row: round(row["Inns_Past_5"] / row["Matches_Past_5"], 3) if row["Matches_Past_5"] > 0 else 0,
    axis=1)
    merged["Matches_Future_5"] = merged["Final Predicted Matches"]


    # --- Predict innings using the innings_model ---
    pred_X = merged[["Matches_Past_5", "Inns_Past_5", "Inns/Match", "Matches_Future_5"]].fillna(0)
    merged["Pred_Innings"] = innings_model.predict(pred_X).round(0)

    # --- Predict runs ---
    merged["Predicted_Runs"] = (merged["Final_Runs_Per_Inning"] * merged["Pred_Innings"]).round(1)
    merged["Total_Runs"] = merged["Career_Runs"] + merged["Predicted_Runs"]


    merged.to_csv("prediction.csv", index = False)
    return merged

print(projection(prediction_players, df_predict))
model_results = pd.DataFrame([
    {
        "Model Component": "Batting Average (Known Matches + Innings)",
        "MAE": round(cv_mean_dict["MAE"], 3),
        "MAPE": round(cv_mean_dict["MAPE"], 3),
        "R²": round(cv_mean_dict["R2_Score"], 3),
        "Within 20%": round(cv_mean_dict["Within_20%"], 3),
        "SMAPE": round(cv_mean_dict["SMAPE"], 3)
    },
    {
        "Model Component": "Match Count Prediction (per Country)",
        "MAE": round(mae_matches, 3),
        "MAPE": round(mape_matches, 3),
        "R²": round(r2_matches, 3),
        "Within 20%": round(within_20pct_matches, 3),
        "SMAPE": round(smape_matches, 3)
    },
    {
        # Results shown below are the printed metrics after running the backtest_innings_prediction function
        "Model Component": "Innings Prediction (per Player)",
        "MAE": 1.30,
        "MAPE": 0.029,
        "R²": 0.990,
        "Within 20%": 0.995,
        "SMAPE": 0.029
    }
])

def show_metrics(model):
    # Set model component as index
    model.set_index("Model Component", inplace=True)

    # Plotting
    ax = model.plot(
        kind='bar',
        figsize=(12, 6),
        width=0.75,
        color=["gold", "orangered", "crimson", "hotpink"]
    )

    plt.title("Model Evaluation Metrics (Excluding MAE)", fontsize=14)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="Metric", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    plt.savefig("model_metrics_comparison.png", dpi=300)
    plt.show()

# Display the DataFrame
model_results.to_csv("scaled_model_results.csv", index = False)
model_results.drop(columns="MAE", inplace = True)
show_metrics(model_results)