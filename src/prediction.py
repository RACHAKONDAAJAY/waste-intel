
import numpy as np
from xgboost import XGBRegressor


def train_demo_model():
    """
    Train a demonstration model using synthetic data.
    Replace this data with real municipal waste records
    when suitable historical data is available.
    """
    rng = np.random.default_rng(42)

    number_of_records = 500

    households = rng.integers(
        100, 1001, number_of_records
    )

    day_of_week = rng.integers(
        0, 7, number_of_records
    )

    weekend = (day_of_week >= 5).astype(int)

    # Artificial example values, not real municipal measurements
    waste_kg = (
        households * 0.35
        + weekend * 20
        + rng.normal(0, 15, number_of_records)
    )

    waste_kg = np.maximum(waste_kg, 0)

    features = np.column_stack([
        households,
        day_of_week
    ])

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(features, waste_kg)

    return model


def predict_waste(model, households, day_of_week):
    features = np.array([[
        households,
        day_of_week
    ]])

    prediction = model.predict(features)[0]

    return max(0.0, float(prediction))
