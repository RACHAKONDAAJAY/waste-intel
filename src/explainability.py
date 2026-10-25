
import numpy as np
import pandas as pd
import shap


def explain_prediction(model, households, day_of_week):
    """
    Explain an XGBoost waste prediction using SHAP.

    households: number of households
    day_of_week: Monday = 0, Sunday = 6
    """

    features = np.array([[
        households,
        day_of_week
    ]], dtype=float)

    feature_names = [
        "Number of households",
        "Day of week (Monday=0)"
    ]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(features)
    contributions = np.asarray(shap_values).reshape(-1)

    base_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP impact (kg)": contributions
    })

    return base_value, explanation
