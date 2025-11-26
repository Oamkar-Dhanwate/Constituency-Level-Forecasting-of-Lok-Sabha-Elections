import joblib
from pathlib import Path

def load_model(model_path: Path):
    """Loads a pre-trained model from a file."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None