# src/model.py
import joblib, pandas as pd

def load_model(path='../models/churn_model_v1.pkl'):
    """Load trained sklearn Pipeline."""
    return joblib.load(path)

def predict_single(model, customer: dict) -> dict:
    """Predict churn for one customer dict."""
    df = pd.DataFrame([customer])
    prob = model.predict_proba(df)[0][1]
    return {'churn_probability': round(float(prob),4),
            'will_churn': bool(prob > 0.5),
            'risk_level': 'HIGH' if prob>0.7
                else 'MEDIUM' if prob>0.4 else 'LOW'}
