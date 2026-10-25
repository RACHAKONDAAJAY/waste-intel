
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(actual_values, predicted_values):
    """
    Evaluate regression predictions.

    actual_values: Real observed values.
    predicted_values: Model predictions.

    Returns MAE, RMSE, and R2.
    """

    actual = np.asarray(actual_values, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_values, dtype=float).reshape(-1)

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Actual and predicted values cannot be empty.")

    if actual.size != predicted.size:
        raise ValueError(
            "Actual and predicted values must have the same length."
        )

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Values must be valid finite numbers.")

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))

    # R² is undefined when there is only one actual observation.
    r2 = (
        float(r2_score(actual, predicted))
        if actual.size >= 2
        else None
    )

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": r2,
    }


if __name__ == "__main__":
    # Example values for testing only — not real municipal data.
    actual = [100, 150, 200, 250, 300]
    predicted = [110, 140, 190, 260, 290]

    metrics = evaluate_regression(actual, predicted)

    print("Regression Evaluation Results")
    print(f"MAE:  {metrics['MAE']:.2f} kg")
    print(f"RMSE: {metrics['RMSE']:.2f} kg")

    if metrics["R2"] is not None:
        print(f"R2:   {metrics['R2']:.4f}")
