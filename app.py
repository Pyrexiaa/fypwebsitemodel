from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from features import name_conversion, value_mappings
from model_paths import (
    BINARY_FEATURES,
    IMPUTATION_MODELS,
    MULTICLASS_FEATURES,
    REGRESSION_FEATURES,
)
from utils import (
    binary_classification,
    get_scale_values,
    impute,
    postprocess_imputed_data,
)

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello() -> dict[str, str]:
    return render_template("index.html")


@app.route("/impute", methods=["POST"])
def impute_data() -> dict[str, str]:
    data = request.json
    print("Received data: ", data)
    # Convert keys
    converted_data = {}
    for k, v in data.items():
        # Get the new key, default to the original key if no mapping exists
        new_key = name_conversion.get(k, k)
        # Add to the converted data dictionary with the original value
        converted_data[new_key] = v

    # Map values
    for k, v in converted_data.items():
        # Check if the key has a specific value mapping - categorical features
        if k in value_mappings:
            # Map the value if it's in the value mapping, else leave as-is
            converted_data[k] = value_mappings[k].get(v, v)
            continue
        converted_data[k] = float(converted_data[k])

    print("Data after key conversion:", converted_data)

    # If certain data is unavailable, impute it
    for feature, _ in IMPUTATION_MODELS.items():
        if feature not in converted_data or converted_data[feature] is None:
            if feature in BINARY_FEATURES:
                feature_type = "binary"
            elif feature in MULTICLASS_FEATURES:
                feature_type = "multiclass"
            elif feature in REGRESSION_FEATURES:
                feature_type = "regression"
            else:
                raise Warning(f"Unknown feature type found: {feature}")
            converted_data[feature] = impute(converted_data, feature, feature_type)

    cleaned_data = postprocess_imputed_data(converted_data)
    return jsonify(cleaned_data)


@app.route("/process", methods=["POST"])
def process_data() -> dict[str, str]:
    data = request.json
    print("Received data: ", data)
    # Convert keys
    converted_data = {}
    for k, v in data.items():
        # Get the new key, default to the original key if no mapping exists
        new_key = name_conversion.get(k, k)
        # Add to the converted data dictionary with the original value
        converted_data[new_key] = v

    # Manually add smoking and diabetes_1 as 0
    converted_data["smoking"] = 0
    converted_data["diabetes_1"] = 0
    print("Converted Data before binary classification: ", converted_data)

    # After imputing, get scale values
    std_and_mean = get_scale_values()
    std = std_and_mean[0]
    mean = std_and_mean[1]

    # Scale it and classify it
    result = binary_classification(converted_data, std, mean)
    print("Results after binary classification: ", result.item())
    return jsonify(result.item())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
