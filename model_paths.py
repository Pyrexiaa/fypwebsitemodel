"""Imputation Models Path.

Excluding smoking and diabetes_1
They are initialized as 0
"""

IMPUTATION_MODELS = {
    "afi": "models/afi",
    "bpd": "models/bpd",
    "cpr": "models/cpr",
    "psv": "models/psv",
    "ute_api": "models/ute_api",
    "ute_ari": "models/ute_ari",
    "af": "models/af",
    "placenta_site": "models/placenta_site",
    "umb_api": "models/umb_api",
    "m_height": "models/m_height",
    "m_weight": "models/m_weight",
    "hypertension_0": "models/hypertension_0",
    "hypertension_1": "models/hypertension_1",
    "diabetes_0": "models/diabetes_0",
    "last_preg_sga": "models/last_preg_sga",
    "last_preg_fgr": "models/last_preg_fgr",
    "last_preg_normal": "models/last_preg_normal",
    "prev_failed_preg": "models/prev_failed_preg",
    "high_risk_pe": "models/high_risk_pe",
}

BINARY_FEATURES = ["hypertension_0", "hypertension_1", "diabetes_0", "diabetes_1", "smoking", "last_preg_sga", "last_preg_fgr", "last_preg_normal", "prev_failed_preg", "high_risk_pe"]
MULTICLASS_FEATURES = ["af", "placenta_site"]
REGRESSION_FEATURES = ["afi", "bpd", "cpr", "ute_api", "psv", "ute_ari", "umb_api", "m_height", "m_weight"]

SCALING_CSV = "scaleValues/value.csv"
NEURAL_NETWORK = "models/generalized_model.pth"
