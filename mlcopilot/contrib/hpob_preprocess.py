import json
import random
from pathlib import Path

import numpy as np
import xgboost as xgb

root_dir = Path(__file__).parent.parent.parent.parent / 'llm-automl'
assert root_dir.exists()

descriptor = json.loads((root_dir / 'hpob/hpob-data/meta-dataset-descriptors.json').read_text())
surrogates_stats = json.loads((root_dir / 'downloads/hpob/saved-surrogates/summary-stats.json').read_text())
flows = json.loads((root_dir / 'hpob/description/flows.json').read_text())
spaces = json.loads((root_dir / 'hpob/description/spaces.json').read_text())
loaded_surrogates = {}

# Type 1 (config): Human-readable config dict. Friendly to LLM. Type can be float, bool, str. Integer is not inferred.
# Type 2 (array): Machine-readable array. Normalized, preproceesed.

def read_data():
    return json.loads((root_dir / 'downloads/hpob/meta-train-dataset.json').read_text())


def array_to_config(array: np.ndarray, search_space_id: str) -> dict:
    description = descriptor[search_space_id]
    variables = descriptor[search_space_id]['variables_order']
    raw_dict = dict(zip(variables, array))
    dtypes = flows[search_space_id]['parameters_meta_info']
    result = {}
    for variable, var_desc in description['variables'].items():
        if variable.endswith('.na'):
            continue
        if 'categories' in var_desc:
            # Categorical variable
            categories = var_desc['categories']
            for category in categories:
                if raw_dict[variable + '.ohe._' + category]:
                    if category != 'INVALID':
                        if category.upper() == 'TRUE':
                            result[variable] = True
                        elif category.upper() == 'FALSE':
                            result[variable] = False
                        else:
                            result[variable] = category
                    break
            else:
                assert False
        else:
            if raw_dict.get(variable + '.na', 0.) == 0:
                intermediate = (var_desc['max'] - var_desc['min']) * raw_dict[variable] + var_desc['min']
                if variable in description['variables_to_apply_log']:
                    # Log-scale variable
                    result[variable] = np.exp(intermediate)
                else:
                    # Linear-scale variable
                    result[variable] = intermediate
                if dtypes[variable]['data_type'] == 'integer':
                    result[variable] = round(result[variable])

    # fill-in default (even with condition)
    result = {
        name: result[name]
        if name in result
        else spaces[search_space_id][name]["default"]
        for name in spaces[search_space_id] if name != "_meta" and
        (name in result or not spaces[search_space_id][name]["condition"])
    }

    result = {
        name: result[name]
        if name in result
        else spaces[search_space_id][name]["default"]
        for name in spaces[search_space_id] if name != "_meta" and
        (name in result or result[spaces[search_space_id][name]["condition"][0]] == spaces[search_space_id][name]["condition"][1])
    }

    # make sure within range
    result = {
        name: max(spaces[search_space_id][name]["low"], min(spaces[search_space_id][name]["high"], result[name]))
        if "low" in spaces[search_space_id][name] and "high" in spaces[search_space_id][name]
        else result[name]
        for name in result
    }

    return result


def config_to_array(config: dict, search_space_id: str) -> np.ndarray:
    result = {}
    description = descriptor[search_space_id]
    for variable in description['variables_order']:
        if variable.endswith('.na'):
            variable_name = variable[:-3]
            result[variable] = int(variable_name not in config)
        elif '.ohe.' in variable:
            variable_name, category = variable.split('.ohe._')
            if category == 'INVALID':
                result[variable] = int(variable_name not in config)
            else:
                result[variable] = int(variable_name in config and category.upper() == str(config[variable_name]).upper())
        else:
            if variable in config:
                stats = description['variables'][variable]
                if variable in description['variables_to_apply_log']:
                    if config[variable] == 0:
                        value = stats['min']
                    else:
                        value = np.log(config[variable])
                else:
                    value = config[variable]
                result[variable] = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                result[variable] = 0

    return np.array([result[v] for v in description['variables_order']])


def random_config(search_space_id: str) -> dict:
    # Generate random array
    description = descriptor[search_space_id]
    variables = description['variables_order']
    var_desc = description['variables']
    array = []
    chosen = dict()
    for variable in variables:
        if '.ohe' in variable:
            variable_name, category = variable.split('.ohe._')
            if variable_name not in chosen:
                categories = var_desc[variable_name]['categories']
                weights = [var_desc[variable_name]['count'][f'{variable_name}.ohe._{category}'] for category in categories]
                chosen[variable_name] = random.choices(categories, weights)[0]
            array.append(chosen[variable_name] == category)
        elif variable.endswith('.na'):
            array.append(1 if random.uniform(0, 1) < var_desc[variable]['mean'] else 0)
        else:
            dtype = flows[search_space_id]['parameters_meta_info'][variable]['data_type']
            log_scale = variable in description['variables_to_apply_log']
            while True:
                value = random.normalvariate(var_desc[variable]['mean'], var_desc[variable]['std'])
                if var_desc[variable]['min'] <= value <= var_desc[variable]['max']:
                    break
            if log_scale:
                value = np.exp(value)
            if dtype == 'integer':
                value = round(value)
            if log_scale:
                if value == 0:
                    value = var_desc[variable]['min']
                else:
                    value = np.log(value)  # Convert back to array type.
            array.append((value - var_desc[variable]['min']) / (var_desc[variable]['max'] - var_desc[variable]['min']))
    return array_to_config(array, search_space_id)


def evaluate_config(config: dict, search_space_id: str, dataset_id: str) -> float:
    if (search_space_id, dataset_id) in loaded_surrogates:
        bst_surrogate = loaded_surrogates[(search_space_id, dataset_id)]
    else:
        bst_surrogate = xgb.Booster()
        surrogate_name = 'surrogate-'+search_space_id+'-'+dataset_id
        bst_surrogate.load_model('downloads/hpob/saved-surrogates/'+surrogate_name+'.json')
        loaded_surrogates[(search_space_id, dataset_id)] = bst_surrogate
    array = config_to_array(config, search_space_id)[None, :]
    x_q = xgb.DMatrix(array)
    new_y = bst_surrogate.predict(x_q)
    return new_y.item() # FIXED  evaluate_config should output acc


def normalize_metric(metric, search_space_id, dataset_id):
    # Continuous evaluation should use min/max from surrogates.
    y_min = surrogates_stats[f'surrogate-{search_space_id}-{dataset_id}']['y_min']
    y_max = surrogates_stats[f'surrogate-{search_space_id}-{dataset_id}']['y_max']
    return np.clip((metric - y_min) / (y_max - y_min), 0, 1) # FIXED bug


def denormalize_metric(metric, search_space_id, dataset_id):
    y_min = surrogates_stats[f'surrogate-{search_space_id}-{dataset_id}']['y_min']
    y_max = surrogates_stats[f'surrogate-{search_space_id}-{dataset_id}']['y_max']
    return metric * (y_max - y_min) + y_min


def prettify_config(config):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, float):
            # Break down v = a * 10^b
            a = v
            b = 0
            while a < 1:
                a *= 10
                b -= 1
            while a >= 10:
                a /= 10
                b += 1

            valid_numbers = [
                1.0, 1.2, 1.6, 2.0,
                2.4, 3.0, 3.2,
                4.0, 5.0, 6.0,
                6.4, 7.0, 8.0,
                9.0,
            ]
            # Find the number closest to a
            closest_number = min(valid_numbers, key=lambda x: abs(x - a))
            new_config[k] = closest_number * 10 ** b
        else:
            new_config[k] = v
    return new_config
