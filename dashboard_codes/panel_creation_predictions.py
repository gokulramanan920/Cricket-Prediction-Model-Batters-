"""
Gokul Ramanan and Ahmad Shaikh
panel_creation.py
5/26/2025
Description: Creates the panel dashboard of the top 15
international run scorers for cricket.
"""

import panel as pn
from runs_api import RUNSAPI
import plotly.graph_objects as go
import plotly.colors as pc
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd

df_one = pd.read_csv("prediction_players.csv")
df_two = pd.read_csv("prediction.csv")

df_pred_insert = pd.DataFrame({
    "Year": 2030,
    "Country": df_two["Country"],
    "Name": df_two["Name"],
    "Format": df_two["Format"],
    "Mat": round(df_two["Pred_Innings"] / df_two["Inns/Match"], 0),
    "Inns": df_two["Pred_Innings"].round(0).astype(int),
    "NO": 0,  # Assuming none are not-outs for simplicity
    "Runs": df_two["Predicted_Runs"].round(1),
    "HS": None,
    "Avg": df_two["Final_Runs_Per_Inning"].round(2),
    "BF": None,
    "SR": None,
    "100s": 0,
    "50s": 0,
    "0s": 0,
    "4s": 0,
    "6s": 0
})

# Concatenate the new rows to the original df_one
df_combined = pd.concat([df_one, df_pred_insert], ignore_index=True)
df_combined["Is_Prediction"] = df_combined["Year"] == 2030

df_combined.to_csv("add_preds.csv", index = False)

print(df_combined)

# Loads javascript dependencies and configures Panel (required)
pn.extension()

# WIDGET DECLARATIONS
api = RUNSAPI()
api.load_runs("add_preds.csv")

# Search Widgets
format_select = pn.widgets.CheckBoxGroup(name="Format", options=["test", "odi", "t20i"], value = ["test", "odi", "t20i"])
country_select = pn.widgets.MultiSelect(name="Country", options=sorted(api.runs["Country"].unique().tolist()), size=6)
year_slider = pn.widgets.IntRangeSlider(name="Year Range", start=api.runs["Year"].min(), end=api.runs["Year"].max(), step=1)
top_n_slider = pn.widgets.IntSlider(name="Top N Players", start=1, end=17, value=17)
player_select = pn.widgets.MultiChoice(
    name="Select Players",
    options=sorted(api.runs["Name"].unique().tolist()),
    placeholder="Choose players to compare (optional)..."
)
career_length_slider = pn.widgets.IntRangeSlider(
    name="Career Length (Years)",
    start=1,
    end=25,
    step=1,
    value=(1, 25)
)
all_formats_toggle = pn.widgets.Checkbox(
    name="Only Include Players in All 3 Formats",
    value=False
)

# Plotting widgets
width = pn.widgets.IntSlider(name = 'Width', start = 250, end = 2000, step = 250, value = 1500)
height = pn.widgets.IntSlider(name = 'Height', start = 200, end = 2500, step = 100, value = 800)
color_by = pn.widgets.Select(
    name="Color By",
    options=["Name", "Country", "Debut Bin"],
    value="Name"
)
career_align_toggle = pn.widgets.Checkbox(
    name="Align Careers to Year 0",
    value=False
)
detailed_tooltip = pn.widgets.Checkbox(
    name="Show Detailed Hover Info",
    value=True
)
plot_metric_select = pn.widgets.RadioButtonGroup(
    name="Plot Metric",
    options=["Cumulative Runs", "Cumulative Batting Average"],
    button_type="primary",
    value="Cumulative Runs"
)

theme_map = {
    "white": "plotly_white",
    "dark": "plotly_dark",
    "gray1": "ggplot2",
    "gray2": "seaborn",
    "white2": "simple_white"
}

theme_select = pn.widgets.RadioButtonGroup(
    name="Plot Theme",
    options=list(theme_map.keys()),
    button_type="success",
    value="dark"
)

# CALLBACK FUNCTIONS
def get_plot(format_select, country_select, year_slider, top_n_slider, player_select, career_length_slider, width,
             height, theme_select_value, plot_metric_select_value, color_by_value, detailed_tooltip_value,
             align_career_value, all_formats_toggle):
    """
        Generate a cumulative runs line chart based on filter selections.

        Parameters:
            format_select (list): Selected match formats (e.g., ['odi', 'test']).
            country_select (list): List of selected countries.
            year_slider (list): List of [start_year, end_year] to filter data by year.
            top_n_slider (int): Number of top players to include based on total runs.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.
            theme_select_value (str): Theme name corresponding to Plotly templates.

        Returns:
            panel.pane.Plotly or panel.pane.Markdown: Plotly pane if data exists,
            otherwise a message pane.
        """
    if plot_metric_select_value == "Cumulative Runs":
        y_col = "cumulative_format_runs"
        hover = "Runs"
        ranking_metric = "Runs"
    elif plot_metric_select_value == "Cumulative Batting Average":
        y_col = "cumulative_format_average"
        hover = "Average"
        ranking_metric = "Average"

    df = api.apply_filters(formats=format_select, countries=country_select, year_range=year_slider,
                           top_n_players=top_n_slider, player_select_value = player_select, ranking_metric=ranking_metric,
                           career_length_slider = career_length_slider, only_all_formats = all_formats_toggle)

    if plot_metric_select_value == "Cumulative Strike Rate":
        df = df[~df["Name"].isin(["Sachin Tendulkar", "Allan Border", "Javed Miandad", "Desmond Haynes",
                                  "Aravinda de Silva", "Mohammed Azharuddin", "Viv Richards", "Marvin Atapattu",
                                  "Sunil Gavaskar", "Saleem Malik", "Gordon Greenidge", "Arjuna Ranatunga",
                                  "Richie Richardson", ])]
        df = df[df["cumulative_SR"].notnull()]

    if df.empty:
        return pn.pane.Markdown("### No data for selected filters.", width=700)

    fig = go.Figure()

    grouped = df.groupby("Name")

    # Choose a color palette (20 vibrant colors)
    color_palette = (
            pc.qualitative.Set3 + pc.qualitative.Set2 +
            pc.qualitative.Bold + pc.qualitative.Pastel +
            pc.qualitative.Dark2 + pc.qualitative.Safe
    )

    # Get unique labels from your dataframe
    unique_labels = df[color_by_value].unique()
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(sorted(unique_labels))}

    x_col = "Career Year" if align_career_value else "Year"
    x_axis_title = "Career Year" if align_career_value else "Year"

    for name, group in grouped:
        color_label = group[color_by_value].iloc[0]
        color = color_map.get(color_label, "#000000")

        custom_data = group[["cumulative_innings", "cumulative_matches", "cumulative_100s",
                             "cumulative_50s"]].values

        if detailed_tooltip_value:
            hovertemplate = (
                f"<b>{name}</b><br>"
                "Year: %{x}<br>"
                f"{hover}: %{{y}}<br>"
                "Innings: %{customdata[0]}<br>"
                "Matches: %{customdata[1]}<br>"
                "100s: %{customdata[2]}<br>"
                "50s: %{customdata[3]}<br>"
                f"{color_by.name}: {color_label}<extra></extra>"
            )
        else:
            hovertemplate = (
                f"<b>{name}</b><br>"
                f"Year: %{{x}}<br>"
                f"{hover}: %{{y}}<br>"
                f"{color_by.name}: {color_label}<extra></extra>"
            )

        historical = group[group["Is_Prediction"] == False]
        predicted = group[group["Is_Prediction"] == True]

        # Historical (solid)
        fig.add_trace(go.Scatter(
            x=historical[x_col],
            y=historical[y_col],
            mode='lines+markers',
            name=name,
            customdata=historical[["cumulative_innings", "cumulative_matches", "cumulative_100s", "cumulative_50s"]],
            line=dict(color=color),
            legendgroup=color_label,
            hovertemplate=hovertemplate
        ))

        # Prediction (dot)
        if not predicted.empty:
            fig.add_trace(go.Scatter(
                x=predicted[x_col],
                y=predicted[y_col],
                mode='markers',
                name=f"{name} (Predicted)",
                customdata=predicted[["cumulative_innings", "cumulative_matches", "cumulative_100s", "cumulative_50s"]],
                line=dict(color=color, dash='dot'),
                marker=dict(size=10, symbol='diamond'),
                legendgroup=color_label,
                hovertemplate=hovertemplate.replace("<b>", "<b>[Predicted] ")
            ))

    if align_career_value:

        curve_df = df[[x_col, y_col]].dropna()

        if not curve_df.empty:
            smoothed = lowess(endog=curve_df[y_col], exog=curve_df[x_col], frac=0.2)

            fig.add_trace(go.Scatter(
                x=smoothed[:, 0],
                y=smoothed[:, 1],
                mode='lines',
                name="Best Fit Curve",
                line=dict(width=4, color='black', dash='dot'),
                hoverinfo='skip',
                showlegend=True
            ))

    fig.update_layout(
        title=f"Cumulative International {hover} Over Time",
        xaxis_title=x_axis_title,
        yaxis_title=f"Cumulative {hover}",
        width=width,
        height=height,
        showlegend=True,
        template= theme_map[theme_select_value]
    )

    return pn.pane.Plotly(fig)

def get_catalog(format_select, country_select, year_slider, top_n_slider,player_select,career_length_slider, all_formats_toggle):
    """
        Generate an interactive data table of filtered run statistics.

        Parameters:
            format_select (list): Selected match formats (e.g., ['t20i']).
            country_select (list): List of selected countries.
            year_slider (tuple): (start_year, end_year) year range filter.
            top_n_slider (int): Number of top run-scorers to include.

        Returns:
            panel.widgets.Tabulator: A paginated and scrollable data table
            of the filtered DataFrame.
    """
    df = api.apply_filters(formats=format_select, countries=country_select, year_range=year_slider,
                           top_n_players=top_n_slider, player_select_value= player_select,
                           career_length_slider = career_length_slider, only_all_formats = all_formats_toggle)
    table = pn.widgets.Tabulator(df, selectable=False, pagination = 'local', page_size = 20)
    return table

# CALLBACK BINDINGS (Connecting widgets to callback functions)
plot = pn.bind(get_plot, format_select, country_select, year_slider, top_n_slider, player_select,
               career_length_slider, width, height, theme_select, plot_metric_select, color_by, detailed_tooltip,
               career_align_toggle, all_formats_toggle)
catalog = pn.bind(get_catalog, format_select, country_select, year_slider, top_n_slider, player_select,
                  career_length_slider, all_formats_toggle)
# DASHBOARD WIDGET CONTAINERS ("CARDS")

card_width = 320

search_card = pn.Card(
    pn.Column(
        # Widget 1
        format_select,
        # Widget 2
        country_select,
        # Widget 3
        year_slider,
        # Widget 4
        top_n_slider,
        # Widget 5
        player_select,
        # Widget 6
        career_length_slider,
        all_formats_toggle

    ),
    title="Search", width=card_width, collapsed=False
)


plot_card = pn.Card(
    pn.Column(
        career_align_toggle,
        # Default Widget
        plot_metric_select,
        # Default Widget 2
        color_by,
        # Default Widget 3
        detailed_tooltip,
        # Widget 1
        width,
        # Widget 2
        height,
        # Widget 3
        theme_select
    ),

    title="Plot", width=card_width, collapsed=False
)


# LAYOUT

layout = pn.template.FastListTemplate(
    title="Top Prospective International Cricket Batters: Predicted Runs by 2030",
    sidebar=[
        search_card,
        plot_card,
    ],
    theme_toggle=False,
    main=[
        pn.Tabs(
            ("Table", catalog),  # Replace None with callback binding
            ("Time Series", plot),  # Replace None with callback binding
            active=1  # Which tab is active by default?
        )

    ],
    header_background='#a93226'

).servable()

layout.show()