# src/preprocessing.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.compose import ColumnTransformer

def get_feature_columns(df: pd.DataFrame):
    """Return (num_cols, cat_cols) tuple."""
    X = df.drop(columns=['Churn','customerID'],
                errors='ignore')
    num = X.select_dtypes(
        include=['int64','float64']).columns.tolist()
    cat = X.select_dtypes(
        include=['object']).columns.tolist()
    return num, cat

def build_preprocessor(num_cols, cat_cols):
    """Build ColumnTransformer pipeline."""
    return ColumnTransformer([
      ('num', StandardScaler(), num_cols),
      ('cat', OneHotEncoder(
          handle_unknown='ignore',
          sparse_output=False), cat_cols)
    ])

