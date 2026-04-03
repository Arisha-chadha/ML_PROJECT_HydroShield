import pandas as pd

df = pd.read_csv("water_quality.csv")
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))
print(df.isnull().sum())
import pandas as pd
import numpy as np

df = pd.read_csv("water_quality.csv")

df = df.fillna(df.median(numeric_only=True))

ph              = df["ph"].values
hardness        = df["Hardness"].values
solids          = df["Solids"].values
chloramines     = df["Chloramines"].values
trihalomethanes = df["Trihalomethanes"].values
turbidity       = df["Turbidity"].values
organic_carbon  = df["Organic_carbon"].values

# Disease Risk
risk_score = (
    ((ph < 6.0) | (ph > 9.0)).astype(int) * 3 +   
    (turbidity > 5.5).astype(int)          * 3 +  
    (solids > 50000).astype(int)            * 2 + 
    (chloramines > 11).astype(int)          * 2 +  
    (trihalomethanes > 100).astype(int)     * 2 +   
    (hardness > 320).astype(int)            * 1 +   
    (organic_carbon > 22).astype(int)       * 1     
)
df["Disease_Risk"] = np.where(risk_score >= 6, 2, np.where(risk_score >= 3, 1, 0))

# Health Risk Score
ph_dev    = np.abs(ph - 7.0) / 7.0
turb_dev  = (turbidity - 1.45) / (6.74 - 1.45)
tds_dev   = (solids - 320) / (61227 - 320)
chl_dev   = (chloramines - 0.35) / (13.1 - 0.35)
thm_dev   = (trihalomethanes - 0.74) / (124.0 - 0.74)
oc_dev    = (organic_carbon - 2.2) / (28.3 - 2.2)

hrs = (
    0.25 * ph_dev  +
    0.25 * turb_dev +
    0.20 * tds_dev  +
    0.15 * chl_dev  +
    0.10 * thm_dev  +
    0.05 * oc_dev
) * 100
df["Health_Risk_Score"] = np.clip(hrs, 0, 100).round(2)

df.to_csv("water_quality.csv", index=False)
print(f"Updated water_quality.csv — {len(df)} rows")
print(df[["Potability", "Disease_Risk", "Health_Risk_Score"]].describe())




import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, mean_absolute_error, r2_score
)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv("water_quality.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

FEATURES = ["ph","Hardness","Solids","Chloramines","Sulfate",
            "Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]

X = df[FEATURES].values
y_pot   = df["Potability"].values
y_risk  = df["Disease_Risk"].values
y_score = df["Health_Risk_Score"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, yp_tr, yp_te = train_test_split(X_scaled, y_pot,   test_size=0.2, random_state=42, stratify=y_pot)
_,    _,    yr_tr, yr_te  = train_test_split(X_scaled, y_risk,  test_size=0.2, random_state=42)
_,    _,    ys_tr, ys_te  = train_test_split(X_scaled, y_score, test_size=0.2, random_state=42)
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
}
results = {}




pot_results = {}
for name, m in clf_models.items():
    m.fit(X_tr, yp_tr)
    y_pred = m.predict(X_te)
    acc  = accuracy_score(yp_te, y_pred)
    f1   = f1_score(yp_te, y_pred, average="weighted")
    auc  = roc_auc_score(yp_te, m.predict_proba(X_te)[:,1]) if hasattr(m, "predict_proba") else None
    cv   = cross_val_score(m, X_scaled, y_pot, cv=StratifiedKFold(5), scoring="accuracy").mean()
    pot_results[name] = {"Accuracy": acc, "F1": f1, "AUC-ROC": auc, "CV-Acc": cv, "model": m}
    print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  CV={cv:.4f}")


risk_results = {}
risk_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
}
for name, m in risk_models.items():
    m.fit(X_tr, yr_tr)
    y_pred = m.predict(X_te)
    acc  = accuracy_score(yr_te, y_pred)
    f1   = f1_score(yr_te, y_pred, average="weighted")
    cv   = cross_val_score(m, X_scaled, y_risk, cv=5, scoring="accuracy").mean()
    risk_results[name] = {"Accuracy": acc, "F1": f1, "CV-Acc": cv, "model": m}
    print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}  CV={cv:.4f}")



best_pot_name  = max(pot_results,  key=lambda k: pot_results[k]["F1"])
best_risk_name = max(risk_results, key=lambda k: risk_results[k]["F1"])
best_pot_model  = pot_results[best_pot_name]["model"]
best_risk_model = risk_results[best_risk_name]["model"]
print(f"\nBest potability model  : {best_pot_name}")
print(f"Best disease risk model: {best_risk_name}")


if hasattr(best_pot_model, "feature_importances_"):
    fi = pd.Series(best_pot_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances (potability):")
    print(fi.round(4))


os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl",        "wb") as f: pickle.dump(scaler,          f)
with open("models/pot_models.pkl",    "wb") as f: pickle.dump(pot_results,     f)
with open("models/risk_models.pkl",   "wb") as f: pickle.dump(risk_results,    f)
with open("models/best_pot.pkl",      "wb") as f: pickle.dump(best_pot_model,  f)
with open("models/best_risk.pkl",     "wb") as f: pickle.dump(best_risk_model, f)
with open("models/features.pkl",      "wb") as f: pickle.dump(FEATURES,        f)
print("\nAll models saved to models/")


import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": list(pot_results.keys()),
    "Accuracy": [pot_results[m]["Accuracy"] for m in pot_results],
    "F1 Score": [pot_results[m]["F1"] for m in pot_results],
    "AUC-ROC": [pot_results[m]["AUC-ROC"] for m in pot_results],
    "CV Accuracy": [pot_results[m]["CV-Acc"] for m in pot_results]
}

metrics_df = pd.DataFrame(data)

metrics_df.set_index("Model")[["Accuracy", "F1 Score", "AUC-ROC", "CV Accuracy"]].plot(
    kind="bar", figsize=(10, 5)
)

plt.title("Model Comparison - Potability Classification")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

fi = pd.Series(best_pot_model.feature_importances_, index=FEATURES)
fi = fi.sort_values()

plt.figure(figsize=(8, 5))
fi.plot(kind="barh")
plt.title("Feature Importance - Potability Prediction")
plt.xlabel("Importance Score")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.show()



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = best_pot_model.predict(X_te)

print("Classification Report:\n")
print(classification_report(
    yp_te,
    y_pred,
    target_names=["Unsafe", "Safe"]
))

cm = confusion_matrix(yp_te, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Unsafe", "Safe"],
            yticklabels=["Unsafe", "Safe"])

plt.title("Confusion Matrix - Potability")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



import matplotlib.pyplot as plt
import numpy as np

labels = ["Low Risk", "Medium Risk", "High Risk"]
counts = [np.sum(y_risk == 0), np.sum(y_risk == 1), np.sum(y_risk == 2)]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, counts)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 10, str(int(h)),
             ha="center", va="bottom")

plt.title("Disease Risk Distribution")
plt.ylabel("Count")
plt.show()



!pip install -q gradio pandas numpy scikit-learn matplotlib

import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


def load_pickle(name):
    if os.path.exists(name):
        with open(name, "rb") as f:
            return pickle.load(f)
    elif os.path.exists("models/" + name):
        with open("models/" + name, "rb") as f:
            return pickle.load(f)
    return None

def load_csv(name):
    if os.path.exists(name):
        return pd.read_csv(name)
    elif os.path.exists("data/" + name):
        return pd.read_csv("data/" + name)
    return None

scaler = load_pickle("scaler.pkl")
pot_model = load_pickle("best_pot.pkl")
df = load_csv("water_quality.csv")

missing = []
if scaler is None:
    missing.append("scaler.pkl")
if pot_model is None:
    missing.append("best_pot.pkl")
if df is None:
    missing.append("water_quality.csv")

if missing:
    raise FileNotFoundError("Missing files: " + ", ".join(missing))


def health_risk_score(ph, solids, chloramines, organic_carbon, trihalomethanes, turbidity):
    ph_dev = abs(ph - 7.0) / 7.0
    turb_dev = (turbidity - 1.45) / (6.74 - 1.45)
    tds_dev = (solids - 320) / (61227 - 320)
    chl_dev = (chloramines - 0.35) / (13.1 - 0.35)
    thm_dev = (trihalomethanes - 0.74) / (124.0 - 0.74)
    oc_dev = (organic_carbon - 2.2) / (28.3 - 2.2)

    score = (
        0.25 * ph_dev +
        0.25 * turb_dev +
        0.20 * tds_dev +
        0.15 * chl_dev +
        0.10 * thm_dev +
        0.05 * oc_dev
    ) * 100

    return float(np.clip(score, 0, 100))


def predict_water(ph, hardness, solids, chloramines, sulfate, conductivity,
                  organic_carbon, trihalomethanes, turbidity):
    try:
        sample = np.array([[
            float(ph), float(hardness), float(solids), float(chloramines), float(sulfate),
            float(conductivity), float(organic_carbon), float(trihalomethanes), float(turbidity)
        ]])

        sample_scaled = scaler.transform(sample)
        pot_pred = int(pot_model.predict(sample_scaled)[0])
        pot_prob = pot_model.predict_proba(sample_scaled)[0]

       
        model_prediction = "Safe to Drink" if pot_pred == 1 else "Not Safe to Drink"

        if solids > 500 or turbidity > 5 or chloramines > 10:
            potability_label = "Not Safe to Drink"
        else:
            potability_label = model_prediction

   
        score = health_risk_score(
            ph, solids, chloramines, organic_carbon, trihalomethanes, turbidity
        )

        # Rule-based override for risk level
        if solids > 5000:
            risk_text = "High Risk"
        elif solids > 500 and score < 33:
            risk_text = "Medium Risk"
        elif score < 33:
            risk_text = "Low Risk"
        elif score < 66:
            risk_text = "Medium Risk"
        else:
            risk_text = "High Risk"

        summary = f"""
### Prediction Result

**Potability:** {potability_label}
**Safe Probability:** {pot_prob[1]*100:.2f}%

**Health Risk Level:** {risk_text}
**Health Risk Score:** {score:.2f} / 100
"""

        check_df = pd.DataFrame({
            "Parameter": ["pH", "Turbidity", "Chloramines", "Trihalomethanes", "TDS"],
            "Your Value": [ph, turbidity, chloramines, trihalomethanes, solids],
            "Safe Limit": ["6.5 - 8.5", "< 5", "< 10", "< 100", "< 500"],
            "Status": [
                "Safe" if 6.5 <= ph <= 8.5 else "Unsafe",
                "Safe" if turbidity < 5 else "Unsafe",
                "Safe" if chloramines < 10 else "Unsafe",
                "Safe" if trihalomethanes < 100 else "Unsafe",
                "Safe" if solids < 500 else "Unsafe",
            ]
        })


        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        ax.set_aspect("equal")
        ax.axis("off")

        zones = [
            (180, 120, "#8BE28B"),
            (120, 60, "#E6DB75"),
            (60, 0, "#EA7B7B")
        ]

        for start, end, color in zones:
            arc = Wedge((0, 0), 1.0, end, start, width=0.22,
                        facecolor=color, edgecolor="none")
            ax.add_patch(arc)

        progress_angle = 180 - (score / 100) * 180
        progress = Wedge((0, 0), 0.92, progress_angle, 180, width=0.10,
                         facecolor="#008A0E", edgecolor="none")
        ax.add_patch(progress)

        for val in [0, 20, 40, 60, 80, 100]:
            a = 180 - (val / 100) * 180
            tx = 1.08 * np.cos(np.radians(a))
            ty = 1.08 * np.sin(np.radians(a))
            ax.text(tx, ty, str(val), ha="center", va="center",
                    fontsize=11, color="#4A4A4A")

        ax.text(0, -0.02, f"{score:.1f}/100",
                ha="center", va="center",
                fontsize=26, fontweight="bold", color="#333333")

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.12, 1.18)
        plt.tight_layout()

        return summary, check_df, fig

    except Exception as e:
        error_text = f"### Error\n`{type(e).__name__}: {str(e)}`"
        empty_df = pd.DataFrame({"Info": ["Prediction failed"]})
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Prediction failed", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return error_text, empty_df, fig


with gr.Blocks(theme=gr.themes.Soft(), title="HydroShield") as demo:
    gr.Markdown("""
    # 💧 HydroShield
    ## Water Potability and Health Risk Assessment System

    Enter water quality parameters and click **Predict**.
    """)

    with gr.Row():
        with gr.Column():
            ph = gr.Slider(0.0, 14.0, value=7.0, step=0.1, label="pH")
            hardness = gr.Slider(47.0, 325.0, value=196.0, step=1.0, label="Hardness")
            solids = gr.Slider(100.0, 62000.0, value=22000.0, step=50.0, label="Solids (TDS)")
        with gr.Column():
            chloramines = gr.Slider(0.35, 13.1, value=7.1, step=0.1, label="Chloramines")
            sulfate = gr.Slider(129.0, 482.0, value=334.0, step=1.0, label="Sulfate")
            conductivity = gr.Slider(181.0, 754.0, value=426.0, step=1.0, label="Conductivity")
        with gr.Column():
            organic_carbon = gr.Slider(2.2, 28.3, value=14.3, step=0.1, label="Organic Carbon")
            trihalomethanes = gr.Slider(0.7, 124.0, value=66.4, step=0.1, label="Trihalomethanes")
            turbidity = gr.Slider(1.4, 6.8, value=4.0, step=0.1, label="Turbidity")

    predict_btn = gr.Button("Predict", variant="primary")

    output_text = gr.Markdown()
    output_table = gr.Dataframe(label="Parameter Check")
    output_plot = gr.Plot(label="HydroShield Risk Meter")

    predict_btn.click(
        fn=predict_water,
        inputs=[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity
        ],
        outputs=[output_text, output_table, output_plot]
    )

demo.launch(share=True, debug=True)

