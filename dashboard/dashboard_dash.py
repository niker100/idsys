import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# csv einlesen
df = pd.read_csv("metriken.csv", parse_dates=["datum"])

# app erstellen
app = dash.Dash(__name__)
app.title = "Metrik Dashboard"

# layout
app.layout = html.Div([
    html.Div([
        html.H1("📊 Umsatz-Dashboard", style={
            "textAlign": "center",
            "marginBottom": "30px",
            "color": "#333",
            "fontFamily": "Arial, sans-serif"
        }),

        html.Div([
            html.Label("🌍 Region auswählen:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="region_dropdown",
                options=[{"label": r, "value": r} for r in sorted(df["region"].unique())],
                value=df["region"].unique()[0],
                clearable=False,
                style={"marginBottom": "20px"}
            ),

            html.Label("📦 Kategorien auswählen:", style={"fontWeight": "bold"}),
            dcc.Checklist(
                id="kategorie_checklist",
                options=[{"label": k, "value": k} for k in sorted(df["kategorie"].unique())],
                value=list(df["kategorie"].unique()),
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginBottom": "30px"}
            )
        ], style={
            "width": "100%",
            "maxWidth": "600px",
            "margin": "0 auto",
            "padding": "20px",
            "border": "1px solid #ccc",
            "borderRadius": "10px",
            "backgroundColor": "#f9f9f9",
            "boxShadow": "2px 2px 10px rgba(0,0,0,0.05)"
        }),

        dcc.Graph(id="umsatz_plot", style={"marginTop": "40px"})
    ], style={"padding": "40px"})
])

# callback
@app.callback(
    Output("umsatz_plot", "figure"),
    [
        Input("region_dropdown", "value"),
        Input("kategorie_checklist", "value")
    ]
)
def update_figure(region, kategorien):
    gefiltert = df[
        (df["region"] == region) &
        (df["kategorie"].isin(kategorien))
    ]

    if gefiltert.empty:
        fig = px.line(title="Keine Daten verfügbar")
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Keine Daten für diese Auswahl.",
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 18}
            }]
        )
        return fig

    diagramm = px.line(
        gefiltert,
        x="datum",
        y="umsatz",
        color="kategorie",
        title=f"Umsatz in Region {region}",
        markers=True
    )

    diagramm.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"family": "Arial", "color": "#333"},
        title={"x": 0.5, "xanchor": "center"},
        margin={"l": 40, "r": 20, "t": 60, "b": 40}
    )

    return diagramm

# server starten
if __name__ == "__main__":
    app.run(debug=True)