import shap
import pandas as pd


def explain_prediction(model, X_row, background_df, feature_columns):
    """
    Computes SHAP values for a single model's prediction.

    background_df is a small reference sample (recent history for the
    city) that SHAP uses as a baseline to measure how far each feature
    pushed the prediction away from a "typical" value.

    Picks the right explainer for the model type: LinearExplainer for
    Ridge, TreeExplainer for Random Forest, KernelExplainer for anything
    else (currently the neural net, wrapped by predict.NeuralNetModel).
    Using LinearExplainer on a tree model throws immediately, since it
    needs a .coef_ attribute that only linear models have. KernelExplainer
    is slower since it treats the model as a black box, but it's the only
    option that works without assuming linear or tree structure, and a
    single live prediction on a small background sample stays fast enough
    for the dashboard.

    Returns a Series of feature -> contribution, sorted by absolute
    impact so the biggest drivers show up first.
    """
    background = background_df[feature_columns].fillna(
        background_df[feature_columns].mean()
    )

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
    elif hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, background)

    shap_values = explainer(X_row[feature_columns])

    contributions = pd.Series(shap_values.values[0], index=feature_columns)
    return contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    )