import shap
import pandas as pd


def explain_prediction(model, X_row, background_df, feature_columns):
    """
    Computes SHAP values for a single Ridge model prediction.

    background_df is a small reference sample (recent history for the
    city) that SHAP uses as a baseline to measure how far each feature
    pushed the prediction away from a "typical" value.

    Returns a Series of feature -> contribution, sorted by absolute
    impact so the biggest drivers show up first.
    """
    background = background_df[feature_columns].fillna(
        background_df[feature_columns].mean()
    )

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(X_row[feature_columns])

    contributions = pd.Series(shap_values.values[0], index=feature_columns)
    return contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    )