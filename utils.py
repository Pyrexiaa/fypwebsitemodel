import ast

import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np

from model_architecture import FNNClassifierTri3
from model_paths import IMPUTATION_MODELS, NEURAL_NETWORK, SCALING_CSV
from features import name_conversion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBABILITY_THRESHOLD = 0.6


def impute(inputs: dict, feature: str, model_type: str) -> float:
    input_features = [
        [
            inputs["m_age"],
            inputs["gender"],
            inputs["hc"],
            inputs["ac"],
            inputs["fl"],
            inputs["efw"],
            inputs["ga"],
        ]
    ]

    # Initialize model based on type
    if model_type == "binary":
        model = CatBoostClassifier()
    elif model_type == "multiclass":
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            auto_class_weights="Balanced",
        )
    elif model_type == "regression":
        model = CatBoostRegressor(verbose=0)
    else:
        raise ValueError("Model type can only be binary, multiclass, or regression")

    # Load the pre-trained model
    model_path = IMPUTATION_MODELS[feature]
    model.load_model(model_path)

    # Use Pool to handle categorical features
    print("Input Features for catboost: ", input_features)
    pool = Pool(data=input_features, cat_features=[1])
    prediction = model.predict(pool, verbose=False)
    return prediction[0]


def postprocess_imputed_data(imputed_data):
    # Reverse the name_conversion mapping
    reversed_mapping = {v: k for k, v in name_conversion.items()}

    return {
        # For each key-value pair in the original data
        reversed_mapping.get(key, key): (
            value.item()
            if isinstance(value, np.ndarray) and value.size == 1
            else int(value)
            if isinstance(value, np.int64)
            else float(value)
            if isinstance(value, np.float64)
            else value
        )
        for key, value in imputed_data.items()
    }


def get_scale_values() -> list:
    scaling_numbers = pd.read_csv(SCALING_CSV)
    scaling_fold = 1
    std_or_min = scaling_numbers.loc[scaling_fold, "std_or_min"]
    std_or_min = ast.literal_eval(std_or_min)
    mean_or_max = scaling_numbers.loc[scaling_fold, "mean_or_max"]
    mean_or_max = ast.literal_eval(mean_or_max)
    return [std_or_min, mean_or_max]


def binary_classification(inputs: dict, std_or_min: float, mean_or_max: float) -> float:
    desired_feature_sequence = [
        "gender",
        "placenta_site",
        "af",
        "hypertension_0",
        "diabetes_0",
        "diabetes_1",
        "smoking",
        "prev_failed_preg",
        "high_risk_pe",
        "ga",
        "bpd",
        "hc",
        "ac",
        "fl",
        "afi",
        "ute_ari",
        "ute_api",
        "m_age",
        "cpr",
        "psv",
        "efw",
        "umb_api",
        "m_height",
        "m_weight",
    ]

    categorical_features = [
        "gender",
        "placenta_site",
        "af",
        "hypertension_0",
        "diabetes_0",
        "diabetes_1",
        "smoking",
        "prev_failed_preg",
        "high_risk_pe",
    ]

    nn_input = []
    added_features = []

    for feature in desired_feature_sequence:
        for key, value in inputs.items():
            if key == feature:
                nn_input.append(value)
                added_features.append(feature)  # Track successfully added features
                break
        else:
            # If the loop does not break, it means the feature was not added
            print("Feature not added: ", feature)

    # Identify features that were not added
    not_added_features = [feature for feature in desired_feature_sequence if feature not in added_features]
    print("Features not added to nn_input:", not_added_features)

    print("NN Input: ", len(nn_input))
    print("Mean or max: ", mean_or_max)
    print("Std or min: ", std_or_min)

    # Scale Value
    scaled_input = []
    for i, col in enumerate(nn_input):
        if i < len(categorical_features):
            scale_value = int(col)
        else:
            mean_or_max_key = list(mean_or_max)[i - len(categorical_features)]
            std_or_min_key = list(std_or_min)[i - len(categorical_features)]
            scale_value = round(
                ((col - mean_or_max[mean_or_max_key]) / std_or_min[std_or_min_key]), 4
            )
        scaled_input.append(scale_value)

    input_size = len(nn_input)
    net = FNNClassifierTri3(input_size, 0.20, 4)
    net.load_state_dict(torch.load(NEURAL_NETWORK, map_location=torch.device("cpu")))
    net.to(DEVICE)

    # Predict the value
    scaled_input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=DEVICE)
    net.eval()
    out = net(scaled_input_tensor)
    return torch.where(
        out.data.sigmoid() > PROBABILITY_THRESHOLD,
        torch.tensor(1, device=DEVICE),
        torch.tensor(0, device=DEVICE),
    )
