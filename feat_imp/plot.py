import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pdpbox import pdp
from pdpbox.info_plot_utils import _prepare_info_plot_data
from pdpbox.utils import _check_feature


def plot_features_importance(
    var_list: list, importances: np.array, n_feat: int = None
) -> go.Figure:
    """
    Plot the features importance barplot.

    Parameters
    ----------
    var_list: list,
        data containing colnames used in the model.

    importances: numpy array,
        list of feature importances

    n_feat: int,
        number of features to plot
    """
    indices = np.argsort(importances)

    if n_feat is None:
        x = importances[indices]
        y = var_list[indices]
        title = "Feature Importance"
    else:
        x = importances[indices][:n_feat]
        y = var_list[indices][:n_feat]
        title = "Top {} Features Importance".format(n_feat)

    fig = go.Figure(
        go.Bar(
            x=importances[indices][:n_feat],
            y=var_list[indices][:n_feat],
            orientation="h",
            marker_color="rgb(46, 186, 175)",
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Percentage of total contribution",
        yaxis_title="Features",
        xaxis=dict(tickformat=".0%"),
        autosize=False,
        width=800,
        height=500,
        plot_bgcolor="rgb(240, 240, 240)",
    )

    return fig.show()


def plot_pdp(
    X: pd.DataFrame,
    y: np.array,
    model,
    var_name: str,
    target_name: str,
    num_grid_points: int = 10,
) -> go.Figure:

    ## Concat target column
    df = pd.concat([X, pd.DataFrame({target_name: y})], axis=1)

    ## Compute mean target per bin
    data_x, _, summary_df, _, _, _ = _prepare_info_plot_data(
        feature=var_name,
        feature_type=_check_feature(var_name, df),
        data=df,
        num_grid_points=num_grid_points,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        show_percentile=False,
        show_outliers=False,
        endpoint=True,
    )

    target_line = (
        data_x.groupby("x", as_index=False)
        .agg({target_name: "mean"})
        .sort_values("x", ascending=True)
    )
    summary_df = summary_df.merge(target_line, on="x", how="outer")[
        ["display_column", target_name]
    ].rename({"display_column": var_name, target_name: "mean_target"}, axis=1)

    ## Compute Partial dependence plot
    pdp_array = pdp.pdp_isolate(
        model=model,
        dataset=X,
        model_features=X.columns,
        feature=var_name,
        num_grid_points=num_grid_points,
    ).pdp
    pdp_array = 0.5 * (pdp_array[1:] + pdp_array[:-1])

    ## Merge with summary_df
    summary_df["pdp"] = pdp_array

    ## Plot figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary_df[var_name],
            y=summary_df["mean_target"],
            mode="lines",
            name="mean_target",
            line=dict(color="rgb(46, 186, 175)", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df[var_name],
            y=summary_df["pdp"],
            mode="lines",
            name="partial_dependence_plot",
            line=dict(color="rgb(255, 161, 105)", width=4),
        )
    )

    title = "Mean conversion rate and Partial dependence plot"
    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        legend=dict(x=0.65, y=0.9),
        xaxis_title=var_name,
        yaxis_title=target_name,
        yaxis_range=[0, 0.25],
        autosize=False,
        width=800,
        height=500,
        plot_bgcolor="rgb(240, 240, 240)",
    )

    return fig.show()
