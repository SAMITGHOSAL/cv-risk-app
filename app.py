import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
from catboost import Pool

# -------------------------
# Initialize Dash
# -------------------------

app = dash.Dash(__name__)
server = app.server

# -------------------------
# Load Machine Learning Model
# -------------------------

with open("catboost.pickle", "rb") as f:
    model = pickle.load(f)

# -------------------------
# App Layout
# -------------------------

app.layout = html.Div([

    html.H1("Rapid Cardiovascular Risk Screening (Truncated Model)"),

    html.Br(),

    html.Label("Age"),
    dcc.Input(id="age", type="number", placeholder="Enter age"),

    html.Br(),
    html.Br(),

    html.Label("Sex"),
    dcc.Dropdown(
        id="sex",
        options=[
            {"label": "Male", "value": 1},
            {"label": "Female", "value": 0}
        ],
        value=1
    ),

    html.Br(),

    html.Label("BMI"),
    dcc.Input(id="bmi", type="number", placeholder="BMI"),

    html.Br(),
    html.Br(),

    html.Label("Smoking Status"),
    dcc.Dropdown(
        id="smoking",
        options=[
            {"label": "Non-smoker", "value": 0},
            {"label": "Smoker", "value": 1}
        ],
        value=0
    ),

    html.Br(),

    html.Label("Systolic Blood Pressure"),
    dcc.Input(id="sbp", type="number", placeholder="SBP"),

    html.Br(),
    html.Br(),

    html.Label("Total Cholesterol"),
    dcc.Input(id="chol", type="number", placeholder="Total Cholesterol"),

    html.Br(),
    html.Br(),

    html.Label("HDL Cholesterol"),
    dcc.Input(id="hdl", type="number", placeholder="HDL"),

    html.Br(),
    html.Br(),

    html.Label("Medical Conditions"),

    dcc.Checklist(
        id="conditions",
        options=[
            {"label": "BP Treatment", "value": "bp_treat"},
            {"label": "Angina / Heart Attack", "value": "angina"},
            {"label": "Chronic Kidney Disease", "value": "ckd"}
        ]
    ),

    html.Br(),

    html.Button("Calculate Risk", id="predict", n_clicks=0),

    html.Br(),
    html.Br(),

    html.Div(id="output"),

    html.Br(),

    html.Hr(),

    html.P(
        "This truncated model is a rapid screening tool and does not replace the full QRISK3 assessment."
    )

])

# -------------------------
# Prediction Function
# -------------------------

@app.callback(
    Output("output", "children"),
    Input("predict", "n_clicks"),
    State("age", "value"),
    State("sex", "value"),
    State("bmi", "value"),
    State("smoking", "value"),
    State("sbp", "value"),
    State("chol", "value"),
    State("hdl", "value"),
    State("conditions", "value")
)

def predict_risk(n_clicks, age, sex, bmi, smoking, sbp, chol, hdl, conditions):

    if n_clicks == 0:
        return ""

    if None in [age, sex, bmi, smoking, sbp, chol, hdl]:
        return html.Div("Please fill all fields", style={"color": "red"})

    if conditions is None:
        conditions = []

    try:

        chol_ratio = chol / hdl

        bp_treat = 1 if "bp_treat" in conditions else 0
        angina = 1 if "angina" in conditions else 0
        ckd = 1 if "ckd" in conditions else 0

        df = pd.DataFrame([[

    float(age),
    float(sex),
    float(bmi),
    float(smoking),
    float(sbp),
    float(bp_treat),
    float(angina),
    float(chol),
    float(chol_ratio),
    float(ckd)

]], columns=[
    "Age",
    "Sex",
    "BMI",
    "Smoking",
    "SystolicBP",
    "BPTreatment",
    "Angina",
    "TotalCholesterol",
    "CholHDLRatio",
    "CKD"
])
pool = Pool(df)
prob = model.predict_proba(pool)[0][1]
        risk_class = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        color = "red" if prob >= 0.5 else "green"

        return html.Div([
            html.H2(risk_class, style={"color": color}),
            html.H3(f"Predicted Probability: {round(prob*100,1)} %")
        ])

    except Exception as e:

        return html.Div(
            f"Prediction Error: {str(e)}",
            style={"color": "red"}
        )

# -------------------------
# Run Local Server
# -------------------------

if __name__ == "__main__":
    app.run_server(debug=True)