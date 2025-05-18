"""Imputation Models Path.

Excluding smoking and diabetes_1
They are initialized as 0
"""

MODEL_FOLDER_PATH = "latest_models"

IMPUTATION_MODELS = {
    "afi": f"{MODEL_FOLDER_PATH}/afi",
    "bpd": f"{MODEL_FOLDER_PATH}/bpd",
    "cpr": f"{MODEL_FOLDER_PATH}/cpr",
    "psv": f"{MODEL_FOLDER_PATH}/psv",
    "ute_api":f"{MODEL_FOLDER_PATH}/ute_api",
    "ute_ari": f"{MODEL_FOLDER_PATH}/ute_ari",
    "af": f"{MODEL_FOLDER_PATH}/af",
    "placenta_site": f"{MODEL_FOLDER_PATH}/placenta_site",
    "umb_api": f"{MODEL_FOLDER_PATH}/umb_api",
    "m_height": f"{MODEL_FOLDER_PATH}/m_height",
    "m_weight": f"{MODEL_FOLDER_PATH}/m_weight",
    "hypertension_0": f"{MODEL_FOLDER_PATH}/hypertension_0",
    "diabetes_0": f"{MODEL_FOLDER_PATH}/diabetes_0",
    "last_preg_sga": f"{MODEL_FOLDER_PATH}/last_preg_sga",
    "last_preg_fgr": f"{MODEL_FOLDER_PATH}/last_preg_fgr",
    "last_preg_normal": f"{MODEL_FOLDER_PATH}/last_preg_normal",
    "prev_failed_preg": f"{MODEL_FOLDER_PATH}/prev_failed_preg",
    "high_risk_pe": f"{MODEL_FOLDER_PATH}/high_risk_pe",
}

BINARY_FEATURES = ["hypertension_0", "hypertension_1", "diabetes_0", "diabetes_1", "smoking", "last_preg_sga", "last_preg_fgr", "last_preg_normal", "prev_failed_preg", "high_risk_pe"]
MULTICLASS_FEATURES = ["af", "placenta_site"]
REGRESSION_FEATURES = ["afi", "bpd", "cpr", "ute_api", "psv", "ute_ari", "umb_api", "m_height", "m_weight"]

SCALING_CSV = "scaleValues/value.csv"
NEURAL_NETWORK = "models/generalized_model.pth"
LOGISTIC_REGRESSION = F"{MODEL_FOLDER_PATH}/logistic_regression"
