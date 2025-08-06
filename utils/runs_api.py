"""
Gokul Ramanan and Ahmad Shaikh
runs_api.py
5/26/2025
Description: RUNS API for extracting data from the Top_15_International_Run_Scorers.csv
file based on various selection parameters.
"""

import pandas as pd

class RUNSAPI:

    def __init__(self):
        self.runs = None

    def load_runs(self, filename):
        """
            Load and preprocess the dataset from a CSV file.

            Parameters:
                filename (str): Path to the CSV file containing run statistics.

            Processing Steps:
                - Loads data into a pandas DataFrame.
                - Converts the 'Runs' column to numeric, coercing non-numeric
                entries to NaN.
                - Sorts the DataFrame by player name, format, and year for
                accurate cumulative calculations.

            Returns:
                None: Modifies the `self.runs` attribute in-place.
            """
        self.runs = pd.read_csv(filename)

        # Clean: Ensure 'Runs' is numeric
        self.runs["Runs"] = pd.to_numeric(self.runs["Runs"], errors="coerce")
        self.runs["Innings"] = pd.to_numeric(self.runs["Inns"], errors="coerce")
        self.runs["Not Outs"] = pd.to_numeric(self.runs["NO"], errors="coerce")
        self.runs["Balls Faced"] = pd.to_numeric(self.runs["BF"], errors="coerce")
        self.runs["Matches"] = pd.to_numeric(self.runs["Mat"], errors="coerce")
        self.runs["100s"] = pd.to_numeric(self.runs["100s"], errors="coerce")
        self.runs["50s"] = pd.to_numeric(self.runs["50s"], errors="coerce")

        # Clean: Sort for proper cumulative computation
        self.runs = self.runs.sort_values(by=["Name", "Format", "Year"])

    def apply_filters(self, formats=None, countries=None, year_range=None, top_n_players = None, player_select_value = None,
                      ranking_metric="Runs", career_length_slider = None, only_all_formats=False):
        """
            Filter the dataset based on selected formats, countries, year range, and
            top N players by run total.
            Parameters:
                formats (list): List of formats to filter (e.g., ['odi', 'test']).
                countries (list): List of country names to include.
                year_range (list): [start_year, end_year] for filtering.
                top_n_players (int): Keep only top N run-scorers across filtered data.
            Returns:
                pd.DataFrame: Filtered and aggregated DataFrame with cumulative runs.
            """
        df = self.runs.copy()

        if formats:
            df = df[df["Format"].isin(formats)]

        if countries:
            df = df[df["Country"].isin(countries)]

        if year_range:
            start_year, end_year = year_range
            df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

        if career_length_slider:
            min_years, max_years = career_length_slider
            career_lengths = df.groupby("Name")["Year"].nunique()
            valid_players = career_lengths[(career_lengths >= min_years) & (career_lengths <= max_years)].index
            df = df[df["Name"].isin(valid_players)]

        if only_all_formats:
            player_format_counts = self.runs.groupby("Name")["Format"].nunique()
            eligible_players = player_format_counts[player_format_counts == 3].index
            df = df[df["Name"].isin(eligible_players)]

        df = df.groupby(["Name", "Year", "Country"], as_index=False).agg({"Runs": "sum", "Innings": "sum",
                                                                          "Not Outs": "sum", "Balls Faced": "sum",
                                                                          "Matches": "sum", "100s": "sum", "50s": "sum"})
        df = df.sort_values(by=["Name", "Year"])
        df["cumulative_matches"] = df.groupby("Name")["Matches"].cumsum()
        df["cumulative_100s"] = df.groupby("Name")["100s"].cumsum()
        df["cumulative_50s"] = df.groupby("Name")["50s"].cumsum()
        df["cumulative_format_runs"] = df.groupby("Name")["Runs"].cumsum()
        df["cumulative_innings"] = df.groupby("Name")["Innings"].cumsum()
        df["cumulative_NO"] = df.groupby("Name")["Not Outs"].cumsum()
        df["cumulative_format_BF"] = df.groupby("Name")["Balls Faced"].cumsum()

        df["cumulative_format_average"] = df["cumulative_format_runs"] / (df["cumulative_innings"] - df["cumulative_NO"])
        df["cumulative_SR"] = df.apply(
            lambda row: (row["cumulative_format_runs"] / row["cumulative_format_BF"]) * 100
            if pd.notnull(row["cumulative_format_BF"]) and row["cumulative_format_BF"] > 0
            else None,
            axis=1
        )

        df["Is_Prediction"] = df["Year"] == 2030

        df = df.sort_values(by=["Name", "Year"])
        df["Career Year"] = df.groupby("Name").cumcount()
        df.loc[df["Is_Prediction"], "Career Year"] = df.groupby("Name")["Career Year"].transform("max") + 5

        if player_select_value:
            df = df[df["Name"].isin(player_select_value)]
        elif top_n_players:
            if ranking_metric == "Runs":
                metric_series = df.groupby("Name")["Runs"].sum()
            elif ranking_metric == "Average":
                # Use final cumulative average for each player (latest year)
                last_year_df = df.sort_values(by=["Year"]).groupby("Name").tail(1)
                metric_series = last_year_df.set_index("Name")["cumulative_format_average"]
            else:
                raise ValueError(f"Unknown ranking_metric: {ranking_metric}")

            top_players = metric_series.sort_values(ascending=False).head(top_n_players).index
            df = df[df["Name"].isin(top_players)]

        # Calculate debut year for each player
        debut_years = df.groupby("Name")["Year"].min()
        df["Debut Year"] = df["Name"].map(debut_years)

        # Bin debut years into 5-year ranges
        def bin_debut_year(year):
            if pd.isnull(year):
                return "Unknown"
            start = int((year // 5) * 5)
            return f"{start}–{start + 4}"

        df["Debut Bin"] = df["Debut Year"].apply(bin_debut_year)

        return df.reset_index()

def main():
    high_international_runs = RUNSAPI()
    high_international_runs.load_runs('Top_85_International_Run_Scorers.csv')

    filtered_df = high_international_runs.apply_filters()
    print(len(filtered_df["Name"].unique()))


if __name__ == "__main__":
    main()