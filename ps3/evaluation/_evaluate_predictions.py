import numpy as np
import pandas as pd


# create a function that takes the predictions and actuals and inputs, as well as
# some sample weight (in our case exposure), and returns the bias of your
# estimates as deviation from the actual exposure adjusted mean
def evaluate_bias(y_true, y_pred, sample_weight):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    weighted_true_mean = np.average(y_true, weights=sample_weight)
    weighted_pred_mean = np.average(y_pred, weights=sample_weight)
    bias = weighted_pred_mean - weighted_true_mean
    return bias


# evaluate deviance
def evaluate_deviance(y_true, y_pred, sample_weight):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Tweedie deviance for power=1.5
    deviance = 2 * np.sum(
        sample_weight
        * (
            y_true ** (2 - 1.5) / ((1 - 1.5) * (2 - 1.5))
            - y_true * y_pred ** (1 - 1.5) / (1 - 1.5)
            + y_pred ** (2 - 1.5) / (2 - 1.5)
        )
    )
    return deviance


# evaluate MAE and RMSE
def evaluate_mae_rmse(y_true, y_pred, sample_weight):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.average(np.abs(y_true - y_pred), weights=sample_weight)
    rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight))
    return mae, rmse


# function to evaluate all metrics, and Return a dataframe with the names of the metrics as index.
def evaluate_all(y_true, y_pred, sample_weight):
    bias = evaluate_bias(y_true, y_pred, sample_weight)
    deviance = evaluate_deviance(y_true, y_pred, sample_weight)
    mae, rmse = evaluate_mae_rmse(y_true, y_pred, sample_weight)

    results = pd.DataFrame(
        {
            "Metric": ["Bias", "Deviance", "MAE", "RMSE"],
            "Value": [bias, deviance, mae, rmse],
        }
    ).set_index("Metric")

    return results


def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount
