# tests/test_pipeline.py
import pytest, joblib, pandas as pd
import sys
from pathlib import Path

# Finds the absolute path of the directory 2 levels up from this file
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)
#import sys; sys.path.append('..')
from src.model import load_model, predict_single

@pytest.fixture
def model():
    return load_model('models/churn_model_v1.pkl')

@pytest.fixture
def sample_customer():
    return {
        'gender':'Female','SeniorCitizen':0,
        'Partner':'Yes','Dependents':'No',
        'tenure':12,'PhoneService':'Yes',
        'MultipleLines':'No',
        'InternetService':'Fiber optic',
        'OnlineSecurity':'No','OnlineBackup':'No',
        'DeviceProtection':'No','TechSupport':'No',
        'StreamingTV':'Yes','StreamingMovies':'Yes',
        'Contract':'Month-to-month',
        'PaperlessBilling':'Yes',
        'PaymentMethod':'Electronic check',
        'MonthlyCharges':80.0,'TotalCharges':960.0
    }

def test_model_loads(model):
    assert model is not None

def test_prediction_is_probability(model, sample_customer):
    result = predict_single(model, sample_customer)
    assert 0.0 <= result['churn_probability'] <= 1.0

def test_prediction_has_risk_level(model, sample_customer):
    result = predict_single(model, sample_customer)
    assert result['risk_level'] in ['LOW','MEDIUM','HIGH']

# Run with: pytest tests/ -v
