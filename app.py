import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("truncated_model_v1.pkl", "rb"))

# -----------------------------
# Create Dash app
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container([

    html.H2("Rapid Cardiovascular Risk Screening (Truncated Model)"),

    html.Hr(),

    dbc.Row([
        dbc.Col(dbc.Input(id="age", type="number", placeholder="Age (years)")),
        dbc.Col(dbc.Input(id="bmi", type="number", placeholder="BMI")),
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col(dbc.Input(id="sbp", type="number", placeholder="Systolic BP (mmHg)")),
        dbc.Col(dbc.Input(id="chol", type="number", placeholder="Total Cholesterol (mg/dL)")),
        dbc.Col(dbc.Input(id="hdl", type="number", placeholder="HDL Cholesterol (mg/dL)")),
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="sex",
                options=[
                    {"label": "Male", "value": 1},
                    {"label": "Female", "value": 0}
                ],
                placeholder="Sex"
            )
        ),
        dbc.Col(
            dcc.Dropdown(
                id="smoking",
                options=[
                    {"label": "Non-smoker", "value": 0},
                    {"label": "Ex-smoker", "value": 1},
                    {"label": "Light smoker", "value": 2},
                    {"label": "Moderate smoker", "value": 3},
                    {"label": "Heavy smoker", "value": 4},
                ],
                placeholder="Smoking Status"
            )
        ),
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col(
            dcc.Checklist(
                id="conditions",
                options=[
                    {"label": "On BP Treatment", "value": "bp_treat"},
                    {"label": "History of Angina / MI", "value": "angina"},
                    {"label": "Chronic Kidney Disease", "value": "ckd"},
                ],
            )
        )
    ]),

    html.Br(),
    dbc.Button("Calculate Risk", id="predict", color="primary"),
    html.Br(),
    html.Br(),

    html.Div(id="output"),

    html.Hr(),

    html.Small(
        "This truncated model is a rapid screening tool and does not replace the full 22-parameter QRISK3 assessment. "
        "Low-risk classification does not exclude high cardiovascular risk."
    )

], fluid=True)

# -----------------------------
# Prediction Callback
# -----------------------------
@app.callback(
    Output("output", "children"),
    Input("predict", "n_clicks"),
    State("age", "value"),
    State("bmi", "value"),
    State("sbp", "value"),
    State("chol", "value"),
    State("hdl", "value"),
    State("sex", "value"),
    State("smoking", "value"),
    State("conditions", "value"),
)

def predict_risk(n_clicks, age, bmi, sbp, chol, hdl, sex, smoking, conditions):

    if n_clicks is None:
        return ""

    # Basic validation
    if None in [age, bmi, sbp, chol, hdl, sex, smoking]:
        return html.Div("Please fill all required fields.", style={"color": "orange"})

    if hdl == 0:
        return html.Div("HDL cannot be zero.", style={"color": "orange"})

    if conditions is None:
        conditions = []

    chol_ratio = chol / hdl

    bp_treat = 1 if "bp_treat" in conditions else 0
    angina = 1 if "angina" in conditions else 0
    ckd = 1 if "ckd" in conditions else 0

    # Create dataframe matching training feature names EXACTLY
    df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "Smoking": smoking,
        "SystolicBP": sbp,
        "BPTreatment": bp_treat,
        "Angina": angina,
        "TotalCholesterol": chol,
        "CholHDLRatio": chol_ratio,
        "CKD": ckd
    }])

    # Prediction
    prob = model.predict_proba(df)[0][1]
    risk_class = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    color = "red" if prob >= 0.5 else "green"

    return html.Div([
        html.H3(risk_class, style={"color": color}),
        html.H4(f"Predicted Probability: {round(prob * 100, 1)} %")
    ])

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)