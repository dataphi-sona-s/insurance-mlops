import pandas as pd
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLS = [
    "gender",
    "smoker",
    "region",
    "medical_history",
    "family_medical_history",
    "exercise_frequency",
    "occupation",
    "coverage_level",
]

NUMERICAL_COLS = [
    "age",
    "bmi",
    "children",
]


def preprocess(df: pd.DataFrame, encoder=None, training=True):
    df = df.copy()

    # Handle missing values
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")

    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    if training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoded = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        encoded = encoder.transform(df[CATEGORICAL_COLS])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index
    )

    final_df = pd.concat(
        [df[NUMERICAL_COLS], encoded_df],
        axis=1
    )

    return final_df, encoder
