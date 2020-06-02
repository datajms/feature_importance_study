import openturns as ot
import pandas as pd


VAR_CONF_TRAIN = {  # In follinwg comments, conversion_rate is denoted by CR
    "day_of_week": {
        "marg": ot.Uniform(-0.49, 6.49),  # Uniform between 0 and 6
        "corr": 0.01,  # Very weak influence on CR
        "bounds": None,
        "round": 0,
    },
    "price": {
        "marg": ot.LogNormal(5.0, 0.7, 8.0),  # mean is about 50
        "corr": -0.7,  # Negative correlation with CR
        "bounds": [1.0, 500.0],
        "round": 1,
    },
    "ratio_shipping": {
        "marg": ot.Normal(0.2, 0.07),  # mean is 0.3
        "corr": -0.05,  # Negative correlation with CR
        "bounds": [0.05, 0.4],
        "round": 4,
    },
    "shipping_time": {
        "marg": ot.Poisson(3.0),  # mean is 3 days
        "corr": -0.3,  # The longer, the less CR
        "bounds": [1, 14],
        "round": 4,
    },
    "nb_rating": {
        "marg": ot.Geometric(0.02),  # mean is 50
        "corr": 0.2,  # The more the nb of rating, the better trust
        "bounds": None,
        "round": 0,  # Already an integer (casted as double)
    },
    "avg_rating": {
        "marg": ot.Triangular(1.0, 4.0, 5.0),  # mode is 4.0
        "corr": 0.25,  # The better rating, the better the CR
        "bounds": None,
        "round": 2,
    },
    "nb_provider_rating": {
        "marg": ot.Geometric(0.001),  # mean is 1000
        "corr": 0.08,  # Weak positive correlation
        "bounds": None,
        "round": 0,  # Already an integer (casted as double)
    },
    "avg_provider_rating": {
        "marg": ot.Triangular(2.5, 4.0, 4.8),  # mode is 4.0
        "corr": 0.01,  # Weak positive correlation
        "bounds": None,
        "round": 2,
    },
    "has_multipayment": {
        "marg": ot.Bernoulli(0.4),  # assume 40% of yes
        "corr": 0.0,  # Assume no correlation
        "bounds": None,
        "round": 0,
    },
    "conversion_rate": {
        "marg": ot.Exponential(15.0, 0.05),  # mean is ~0.12 (0.05 + 1/15)
        "corr": None,  # Corr not relevant for conversion_rate, because
        # it is the target variable.
        "bounds": None,
        "round": 4,
    },
}

CORR_CONF_TRAIN = {  # Conf file for adding correlations between variables
    ("price", "ratio_shipping"): -0.15,  # The more expensive, the less costly
    # the shipping
}

VAR_CONF_EVAL = VAR_CONF_TRAIN.copy()
CORR_CONF_EVAL = CORR_CONF_TRAIN.copy()

VAR_CONF_TEST = VAR_CONF_TRAIN.copy()
CORR_CONF_TEST = CORR_CONF_TRAIN.copy()

GENERATION_CONF = {
    "train": {"var": VAR_CONF_TRAIN, "corr": CORR_CONF_TRAIN,},
    "eval": {"var": VAR_CONF_EVAL, "corr": CORR_CONF_EVAL,},
    "test": {"var": VAR_CONF_TEST, "corr": CORR_CONF_TEST,},
}


def post_process_generated_dataset(df_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Add shipping_price variable and delete ratio_shipping variables
    Equation is:
    shipping_price = price * ratio_shipping / (1 - ratio_shipping)

    Parameters
    ----------
    df_sample: pd.DataFrame
        input dataframe

    Returns
    -------
    df_output: pd.DataFrame
        Transformed df
    """
    if df_sample["ratio_shipping"].max() >= 1:
        raise ValueError("Ratio_shipping must be strictly smaller than 1.")

    df_output = df_sample.copy()

    df_output["ratio_shipping"] = (
        df_output["price"]
        * df_output["ratio_shipping"]
        / (1 - df_output["ratio_shipping"])
    )

    df_output = df_output.rename(
        lambda x: "shipping_price" if x == "ratio_shipping" else x, axis="columns"
    )

    return df_output
