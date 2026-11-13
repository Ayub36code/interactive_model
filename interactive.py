# In a separate script
from interactive_model import *

# Load model
model, scaler, feature_names, target_names = ModelPersistence.load_model(
    InteractiveClassifier, 'interactive_model.pth', 'model_metadata.json'
)

# Predict single sample
predictor = InteractivePredictor(model, scaler, feature_names, target_names)
result = predictor.predict_single([12.5, 10.3, 8.9, ...])
print(f"Prediction: {result['class_name']}")