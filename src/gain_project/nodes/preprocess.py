import pandas as pd

REQUIRED_COLUMNS = [
    "before_exam_125_Hz",
    "before_exam_250_Hz",
    "before_exam_500_Hz",
    "before_exam_1000_Hz",
    "before_exam_2000_Hz",
    "before_exam_4000_Hz",
    "before_exam_8000_Hz",
    "after_exam_125_Hz",
    "after_exam_250_Hz",
    "after_exam_500_Hz",
    "after_exam_1000_Hz",
    "after_exam_2000_Hz",
    "after_exam_4000_Hz",
    "after_exam_8000_Hz",
]

def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return df


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_numeric = df.copy()
    for col in REQUIRED_COLUMNS:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
    return df_numeric


def detect_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    invalid_mask = (
        df.isna() |
        (df < 0) |
        (df > 120)
    ).any(axis=1)

    invalid_indices = df[invalid_mask].index.to_frame(index=False)
    invalid_indices.columns = ["invalid_row_index"]

    return invalid_indices


def split_valid_invalid(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = (
        ~(
            df.isna() |
            (df < 0) |
            (df > 120)
        ).any(axis=1)
    )

    clean_df = df[valid_mask].reset_index(drop=True)
    return clean_df