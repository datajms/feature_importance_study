import numpy as np
import pandas as pd
import openturns as ot

from .conf_file_generation import GENERATION_CONF, post_process_generated_dataset


def sample_from_conf(
    var_conf: dict, corr_conf: dict, n_sample: int, seed: int = None
) -> pd.DataFrame:
    """
    Generate dataset with n_sample form configuration file var_conf.

    Parameters
    ----------
    var_conf: dict
        Configuration file of the variables (correlations,
        marginal distributions, rounding)
    n_sample: int
        Number of row in output dataset
    seed: int, optional
        Optional seed for replicability

    Outputs
    -------
    df_sample: pd.DataFrame
        dataset generated from conf files
    """

    ## Retrieve target variable
    var_list = list(var_conf.keys())
    target_var = var_list[-1]
    i_target_var = len(var_list) - 1
    assert var_conf[target_var]["corr"] is None  # Make sure that correlation
    # parameter is set to None for the target variable.

    ## Extract var to i_var dict
    var_dict = {}
    for i_var, var in enumerate(var_list):
        var_dict[var] = i_var

    ## Define marginal distributions of each variable
    marginals = []
    for var in var_list:
        marginals.append(var_conf[var]["marg"])

    ## Define correlations with target variable
    R = ot.CorrelationMatrix(len(var_list))
    for i_var, var in enumerate(var_list):
        if var != target_var:
            R[i_var, i_target_var] = var_conf[var]["corr"]

    ## Define correlations within explanatory variables
    for key, value in corr_conf.items():

        i_min = min(var_dict[key[0]], var_dict[key[1]])
        i_max = max(var_dict[key[0]], var_dict[key[1]])

        R[i_min, i_max] = value

    ## Build distribution and sample
    copula = ot.NormalCopula(R)
    distribution = ot.ComposedDistribution(marginals, copula)

    if seed is not None:
        ot.RandomGenerator.SetSeed(seed)

    df_sample = pd.DataFrame(
        np.array(distribution.getSample(n_sample)), columns=var_list
    )

    ## Apply bounds
    for var in var_list:
        if var_conf[var]["bounds"] is not None:
            df_sample[var] = df_sample[var].clip(
                var_conf[var]["bounds"][0], var_conf[var]["bounds"][1]
            )

    ## Applys rounding
    for var in var_list:
        df_sample[var] = df_sample[var].round(var_conf[var]["round"])

    ## Apply post-processinf

    df_sample = post_process_generated_dataset(df_sample)

    return df_sample


def prepare_ML_sets(
    generation_conf: dict, n_sample: int, test_size: float = 0.25, seed: int = None
) -> tuple:
    """
    Generate train, eval and test sets in X, y scikit-learn format.

    Parameters
    ----------
    generation_conf: dict
        Configuration file of dataset
    n_sample: int
        Number of row in output dataset
    test_size: float, optional
        Proportion of test_size. Note that eval_size is set to eval_size
    seed: int, optional

    Returns
    -------
    output: tuple
        tuple of generated datasets with format:
        (X_train, y_train, X_eval, y_eval, X_test, y_test)
    """

    ## Get target_var name
    target_var = list(generation_conf["train"]["var"].keys())[-1]

    steps = ["train", "eval", "test"]
    n_sample_list = [
        int(n_sample * (1 - 2 * test_size)),
        int(n_sample * test_size),
        int(n_sample * test_size),
    ]

    output = []

    for i_step, (step, i_sample) in enumerate(zip(steps, n_sample_list)):
        if seed is None:  # Change seed for each step
            current_seed = None
        else:
            current_seed = seed + i_step

        df_step = sample_from_conf(
            generation_conf[step]["var"],
            generation_conf[step]["corr"],
            i_sample,  #
            seed=current_seed,
        )

        output += [df_step.drop([target_var], axis=1), df_step[target_var]]

    return tuple(output)
