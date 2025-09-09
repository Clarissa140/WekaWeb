import os
import joblib

MODEL_FOLDER = "models/"

def save_model(model, filename):
    filepath = os.path.join(MODEL_FOLDER, filename)
    joblib.dump(model, filepath)

def load_model(filename):
    filepath = os.path.join(MODEL_FOLDER, filename)
    return joblib.load(filepath)

def list_models():
    return [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
