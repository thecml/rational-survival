from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from scipy.integrate import trapezoid
from sklearn.metrics import r2_score
from sksurv.ensemble import RandomSurvivalForest
import shap
import warnings


def shap_tree_explainer(explainer, new_observation, function_type, aggregation_method, timestamps, **kwargs):
    if timestamps is not None:
        warnings.warn(
            "timestamps are ignored for calculation_method = 'treeshap' \n SurvSHAP(t) values are calculated for explainer.model.unique_times_"
        )
    timestamps = explainer.model.unique_times_

    if not isinstance(explainer.model, RandomSurvivalForest):
        raise TypeError("explained model must be of class sksurv.ensemble.RandomSurvivalForest")

    if function_type == "sf":
        start_index = 1
    elif function_type == "chf":
        start_index = 0

    all_function_vals = [f.y for f in explainer.predict(explainer.data, function_type)]
    baseline_fun = np.mean(all_function_vals, axis=0)
    target_fun = explainer.predict(new_observation, function_type)[0].y

    n_estimators = len(explainer.model.estimators_)
    tree_ensemble_model = {"trees": [estimator.tree_ for estimator in explainer.model.estimators_]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        exp = shap.TreeExplainer(tree_ensemble_model, explainer.data, **kwargs)
        res = exp.shap_values(new_observation)

    shap_values = np.vstack(res[start_index::2]).T / n_estimators

    variable_names = explainer.data.columns
    result = prepare_result_df(new_observation, variable_names, shap_values, timestamps, aggregation_method)
    return result, target_fun, baseline_fun, timestamps


def shap_kernel_explainer(explainer, new_observation, function_type, aggregation_method, timestamps, **kwargs):
    target_fun = explainer.predict(new_observation, function_type)[0]
    all_functions = explainer.predict(explainer.data, function_type)

    if timestamps is None:
        target_fun = target_fun.y
        all_functions_vals = [f.y for f in all_functions]
        timestamps = all_functions[0].x
    else:
        target_fun = target_fun(timestamps)
        all_functions_vals = [f(timestamps) for f in all_functions]
    baseline_fun = np.mean(all_functions_vals, axis=0)

    def predict_function(X):
        all_functions = explainer.predict(X, function_type)
        preds = np.array([f(timestamps) for f in all_functions])
        return preds

    # as shap convert pd.DataFrame to np.array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        exp = shap.KernelExplainer(predict_function, explainer.data, **kwargs)
        res = exp.shap_values(new_observation)

    shap_values = np.vstack(res).T

    variable_names = explainer.data.columns
    result = prepare_result_df(new_observation, variable_names, shap_values, timestamps, aggregation_method)
    return result, target_fun, baseline_fun, timestamps


def shap_kernel(
    explainer, new_observation, function_type, aggregation_method, timestamps, max_shap_value_inputs=np.inf
):
    p = new_observation.shape[1]

    # only one new_observation allowed
    target_fun = explainer.predict(new_observation, function_type)[0]
    all_functions = explainer.predict(explainer.data, function_type)

    if timestamps is None:
        target_fun = target_fun.y
        all_functions_vals = [f.y for f in all_functions]
        timestamps = all_functions[0].x
    else:
        target_fun = target_fun(timestamps)
        all_functions_vals = [f(timestamps) for f in all_functions]
    baseline_f = np.mean(all_functions_vals, axis=0)

    if 2**p < max_shap_value_inputs:
        simplified_inputs = [list(z) for z in itertools.product(range(2), repeat=p)]
    else:
        simplified_inputs = list(np.random.randint(0, 2, size=(max_shap_value_inputs, p)))
        print(
            f"Approximate Survival Shapley will sample only {max_shap_value_inputs} values instead of 2**{p} for Exact Shapley"
        )

    kernel_weights = generate_shap_kernel_weights(simplified_inputs, p)
    shap_values, r2 = calculate_shap_values(
        explainer,
        function_type,
        baseline_f,
        explainer.data,
        simplified_inputs,
        kernel_weights,
        new_observation,
        timestamps,
    )

    variable_names = explainer.data.columns
    result = prepare_result_df(new_observation, variable_names, shap_values, timestamps, aggregation_method)
    return result, target_fun, baseline_f, timestamps, r2


def generate_shap_kernel_weights(simplified_inputs, num_variables):
    weights = []
    for coalition_vector in simplified_inputs:
        num_available_variables = np.count_nonzero(coalition_vector)
        if num_available_variables == 0 or num_available_variables == num_variables:
            weights.append(1e9)
        else:
            weights.append(
                (num_variables - 1)
                / (
                    math.comb(num_variables, num_available_variables)
                    * num_available_variables
                    * (num_variables - num_available_variables)
                )
            )
    return weights


def make_prediction_for_simplified_input(model, function_type, data, simplified_inputs, new_observation, timestamps):
    preds = np.zeros((len(simplified_inputs), len(timestamps)))
    for i, mask in enumerate(simplified_inputs):
        X_tmp = pd.DataFrame(np.where(mask, new_observation, data), columns=data.columns)
        preds[i,] = calculate_mean_function(model, function_type, X_tmp, timestamps)
    return preds


def calculate_shap_values(
    model,
    function_type,
    avg_function,
    data,
    simplified_inputs,
    shap_kernel_weights,
    new_observation,
    timestamps,
):
    W = np.diag(shap_kernel_weights)
    X = np.array(simplified_inputs)
    R = np.linalg.inv(X.T @ W @ X) @ (X.T @ W)
    y = (
        make_prediction_for_simplified_input(model, function_type, data, simplified_inputs, new_observation, timestamps)
        - avg_function
    )
    shap_values = R @ y
    y_pred = X @ shap_values
    r2 = [None] * y.shape[1]
    for i in range(y.shape[1]):
        r2[i] = r2_score(y[:, i], y_pred[:, i], sample_weight=shap_kernel_weights)
    return shap_values, r2


def shap_sampling(
    explainer,
    new_observation,
    function_type,
    path,
    B,
    random_state,
    aggregation_method,
    timestamps,
    exact=False,
):
    p = new_observation.shape[1]

    # only one new_observation allowed
    target_fun = explainer.predict(new_observation, function_type)[0]
    all_functions = explainer.predict(explainer.data, function_type)

    if timestamps is None:
        target_fun = target_fun.y
        all_functions_vals = [f.y for f in all_functions]
        timestamps = all_functions[0].x
    else:
        target_fun = target_fun(timestamps)
        all_functions_vals = [f(timestamps) for f in all_functions]
    baseline_f = np.mean(all_functions_vals, axis=0)

    if exact:
        permutations = [list(perm) for perm in itertools.permutations(np.arange(p), p)]
        B = len(permutations)
    else:
        permutations = [None] * B

    np.random.seed(random_state)
    result_list = [
        iterate_paths(
            explainer,
            function_type,
            explainer.data,
            new_observation,
            timestamps,
            p,
            b + 1,
            np.random,
            permutations[b],
        )
        for b in tqdm(range(B))
    ]

    result = pd.concat(result_list)

    if path is not None:
        if isinstance(path, str) and path == "average":
            average_changes = (
                result.groupby("variable_str").mean(numeric_only=True).iloc[:, 2:]
            )  # choose predictions, not variable value and B
            average_changes["aggregated_change"] = aggregate_change(average_changes, aggregation_method, timestamps)
            average_changes = average_changes.sort_values("aggregated_change", ascending=False)
            result_average = result_list[0].set_index("variable_str").reindex(average_changes.index).reset_index()
            result_average = result_average.assign(B=0)
            result_average.iloc[:, 4:] = average_changes.drop("aggregated_change", axis=1).values
            result_average.insert(4, "aggregated_change", average_changes["aggregated_change"].values)
            result = pd.concat((result_average, result), axis=0)
        else:
            tmp = get_single_random_path(
                explainer,
                function_type,
                explainer.data,
                new_observation,
                timestamps,
                path,
                0,
            )
            result = pd.concat((result, tmp))

    return result, target_fun, baseline_f, timestamps


def iterate_paths(model, function_type, data, new_observation, timestamps, p, b, rng, path):
    if path is None:
        random_path = rng.choice(np.arange(p), p, replace=False)
    else:
        random_path = path
    return get_single_random_path(model, function_type, data, new_observation, timestamps, random_path, b)


def get_single_random_path(model, function_type, data, new_observation, timestamps, random_path, b):
    current_data = deepcopy(data)
    yhats = [None] * (len(random_path) + 1)
    yhats[0] = calculate_mean_function(model, function_type, current_data, timestamps)
    for i, candidate in enumerate(random_path):
        current_data.iloc[:, candidate] = new_observation.iloc[0, candidate]
        yhats[i + 1] = calculate_mean_function(model, function_type, current_data, timestamps)

    diffs = np.diff(yhats, axis=0)

    variable_names = data.columns[random_path]

    new_observation_f = new_observation.loc[:, variable_names].apply(lambda x: nice_format(x.iloc[0]))

    result_diffs = pd.DataFrame(diffs.tolist(), columns=[" = ".join(["t", str(time)]) for time in timestamps])
    result_meta = pd.DataFrame(
        {
            "variable_str": [" = ".join(pair) for pair in zip(variable_names, new_observation_f)],
            "variable_name": variable_names,
            "variable_value": new_observation.loc[:, variable_names].values.reshape(
                -1,
            ),
            "B": b,
        }
    )

    return pd.concat([result_meta, result_diffs], axis=1)


def prepare_result_df(new_observation, variable_names, shap_values, timestamps, aggregation_method):
    new_observation_f = new_observation.apply(lambda x: nice_format(x.iloc[0]))
    result_shap = pd.DataFrame(shap_values, columns=[" = ".join(["t", str(time)]) for time in timestamps])
    result_meta = pd.DataFrame(
        {
            "variable_str": [" = ".join(pair) for pair in zip(variable_names, new_observation_f)],
            "variable_name": variable_names,
            "variable_value": new_observation.values.reshape(
                -1,
            ),
            "B": 0,
            "aggregated_change": aggregate_change(result_shap, aggregation_method, timestamps),
        }
    )
    return pd.concat([result_meta, result_shap], axis=1)


def aggregate_change(average_changes, aggregation_method, timestamps):
    if aggregation_method == "sum_of_squares":
        return np.sum(average_changes.values**2, axis=1)
    if aggregation_method == "max_abs":
        return np.max(np.abs(average_changes.values), axis=1)
    if aggregation_method == "mean_abs":
        return np.mean(np.abs(average_changes.values), axis=1)
    if aggregation_method == "integral":
        return trapezoid(np.abs(average_changes.values), timestamps)


def calculate_mean_function(model, function_type, data, timestamps):
    all_functions = model.predict(data, function_type)
    all_function_vals = [f(timestamps) for f in all_functions]
    return np.mean(all_function_vals, axis=0)


def nice_format(x):
    return str(x) if isinstance(x, (str, np.str_)) else str(float(signif(x)))


def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def calculate_risk_table(ticks, event_times, event_ind):
    n_at_risk = []
    n_censored = []
    n_events = []
    for i in ticks:
        n_at_risk.append((event_times > i).sum())
        n_events.append(event_ind[event_times <= i].sum())
        n_censored.append((~event_ind[event_times <= i]).sum())
    return n_at_risk, n_events, n_censored


def check_new_observation(new_observation, explainer):
    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_frame().T
        new_observation_.columns = explainer.data.columns
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 1:
            # make 1D array 2D
            new_observation_ = new_observation_.reshape((1, -1))
        elif new_observation_.ndim > 2:
            raise ValueError("Wrong new_observation dimension")
        elif new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_ = pd.DataFrame(new_observation_)
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_.columns = explainer.data.columns
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_
