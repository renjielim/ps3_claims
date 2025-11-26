# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
if (df["Exposure"] <= 0).any():
    raise ValueError(
        "Exposure must be strictly positive to derive per-unit pure premiums."
    )
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]


# TODO: use your create_sample_split function here
df = create_sample_split(df, id_column="IDpol", training_frac=0.9)

train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "spline",
            SplineTransformer(
                degree=3, n_knots=5, knots="quantile", include_bias=False
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# %%

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

# %%

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

model_pipeline_gbm = Pipeline(
    steps=[("estimate", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))]
)


model_pipeline_gbm.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline_gbm.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline_gbm.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators

cv = GridSearchCV(
    estimator=model_pipeline_gbm,
    param_grid={
        "estimate__learning_rate": [0.01, 0.1, 0.2],
        "estimate__n_estimators": [100, 200, 500],
    },
    scoring="neg_mean_poisson_deviance",
    cv=3,
    n_jobs=-1,
    verbose=1,
)

cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
best_model_unconstrained = cv.best_estimator_
df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
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


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# PS4 EXERCISE 1
# plot average claim amount against BonusMalus, weighted by Exposure

# %%
fig, ax = plt.subplots(figsize=(8, 6))
df_test.groupby("BonusMalus").apply(
    lambda x: np.average(x["pp_t_lgbm"], weights=x["Exposure"])
).plot(ax=ax)
ax.set(
    title="Average Predicted (LGBM) Premium vs BonusMalus",
    xlabel="BonusMalus",
    ylabel="Average Predicted (LGBM) Premium (weighted by Exposure)",
)
plt.plot()

# %%
# run a new prediction with a monotonic constraint on BonusMalus
model_pipeline_gbm_mc = Pipeline(
    steps=[
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                monotone_constraints=[0, 0, 0, 0, 0, 0, 0, 1, 0],
            ),
        )
    ]
)

cv = GridSearchCV(
    estimator=model_pipeline_gbm_mc,
    param_grid={
        "estimate__learning_rate": [0.01, 0.1, 0.2],
        "estimate__n_estimators": [100, 200, 500],
    },
    scoring="neg_mean_poisson_deviance",
    cv=3,
    n_jobs=-1,
    verbose=1,
)

cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)

# %%
print(
    "training loss t_lgbm_mc:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t
        )
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_mc:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%
# plot average claim amount (after constrain) against BonusMalus, weighted by Exposure
fig, ax = plt.subplots(figsize=(8, 6))
df_test.groupby("BonusMalus").apply(
    lambda x: np.average(x["pp_t_lgbm_constrained"], weights=x["Exposure"])
).plot(ax=ax)
ax.set(
    title="Average Predicted (LGBM) constrained Premium vs BonusMalus",
    xlabel="BonusMalus",
    ylabel="Average Predicted (LGBM) Premium (weighted by Exposure)",
)
plt.plot()
# %%
# EXERCISE 2
# Re-fit the best constrained lgbm estimator from the cross-validation and provide the tuples of the test and train dataset to the estimator via eval_set

best_model_constrained = cv.best_estimator_

best_model_constrained.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    estimate__eval_sample_weight=[w_train_t, w_test_t],
)


# %%
import lightgbm as lgb

lgb.plot_metric(best_model.named_steps["estimate"])

# %%

from ps3.evaluation import (
    evaluate_all,
    evaluate_bias,
    evaluate_deviance,
    evaluate_mae_rmse,
)

# %%
# EXERCISE 3
# evaluate constrained lgbm
results_constrained = evaluate_all(y_test_t, df_test["pp_t_lgbm_constrained"], w_test_t)
print("Evaluation metrics for constrained LGBM:")
print(results_constrained)
# %%
# evaluate unconstrained lgbm
results_unconstrained = evaluate_all(y_test_t, df_test["pp_t_lgbm"], w_test_t)
print("Evaluation metrics for unconstrained LGBM:")
print(results_unconstrained)
# %%
# lorenz curve for all models
fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
    ("LGBM Constrained", df_test["pp_t_lgbm_constrained"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()
# EXERCISE 4
# %%
import dalex as dx

explainer_constrained = dx.Explainer(
    best_model_constrained,
    X_train_t,
    y_train_t,
    weights=w_train_t,
    label="LGBM Constrained",
)

explainer_unconstrained = dx.Explainer(
    best_model_unconstrained,
    X_train_t,
    y_train_t,
    weights=w_train_t,
    label="LGBM Unconstrained",
)


# %%
pdp_constrained = explainer_constrained.model_profile()
pdp_unconstrained = explainer_unconstrained.model_profile()
# %%
pdp_constrained.plot()
# %%
pdp_unconstrained.plot()

# it's clear now monotonicity not present here
# %%
pdp_constrained.plot(pdp_unconstrained)


# %%
