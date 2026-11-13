"""
Interactive Classification Model with PyTorch
- Train on scikit-learn dataset
- Save/load model
- Predict on custom user input
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


# ============================================
# 1. MODEL DEFINITION
# ============================================

class InteractiveClassifier(nn.Module):
    """Neural network for interactive classification"""

    def __init__(self, input_size, hidden_sizes=[64, 32, 16], output_size=2, dropout=0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================
# 2. DATA LOADER
# ============================================

class DataManager:
    """Manage dataset loading and preprocessing"""

    @staticmethod
    def load_dataset(dataset_name='breast_cancer', test_size=0.2, random_state=42):
        """Load and preprocess dataset"""

        print(f"\n📊 Loading dataset: {dataset_name}")

        if dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = ['Malignant', 'Benign']
            n_classes = 2

        elif dataset_name == 'iris':
            data = load_iris()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = list(data.target_names)
            n_classes = 3

        elif dataset_name == 'make_classification':
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_classes=2,
                random_state=random_state
            )
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            target_names = ['Class_0', 'Class_1']
            n_classes = 2

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"   - Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"   - Classes: {n_classes} ({target_names})")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': target_names,
            'n_classes': n_classes,
            'input_size': X.shape[1],
            'scaler': scaler
        }


# ============================================
# 3. MODEL TRAINER
# ============================================

class ModelTrainer:
    """Handle model training and evaluation"""

    @staticmethod
    def train(model, X_train, y_train, X_test, y_test,
              epochs=100, batch_size=32, lr=0.001, verbose=True):
        """Train the model"""

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"\n🏋️ Training model for {epochs} epochs...")
        print("-" * 70)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total = y_test.size(0)
                val_correct = (val_predicted == y_test).sum().item()

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            # Store history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss / len(train_loader):.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}%")

        print("-" * 70)
        print(f"✅ Training complete! Final validation accuracy: {val_acc:.2f}%")

        return history

    @staticmethod
    def evaluate(model, X_test, y_test, target_names):
        """Evaluate the model and return metrics"""

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)

            # Convert to numpy for sklearn metrics
            y_true = y_test.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=target_names)
            conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'true_labels': y_true
        }

    @staticmethod
    def plot_results(history, metrics, target_names, dataset_name):
        """Create visualization plots"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model Performance - {dataset_name}', fontsize=16, fontweight='bold')

        # 1. Training curves
        ax1 = axes[0, 0]
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Confusion Matrix
        ax3 = axes[1, 0]
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=target_names, yticklabels=target_names)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        ax3.set_title('Confusion Matrix')

        # 4. Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"📊 Evaluation Summary\n\n"
        summary += f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n\n"
        summary += f"Detailed Report:\n{metrics['classification_report']}"

        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.tight_layout()
        plt.savefig(f'model_results_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()


# ============================================
# 4. PREDICTOR (INTERACTIVE)
# ============================================

class InteractivePredictor:
    """Handle interactive predictions"""

    def __init__(self, model, scaler, feature_names, target_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.target_names = target_names
        self.model.eval()

    def predict_single(self, input_values):
        """Predict on a single input"""

        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)

        # Scale the input
        input_scaled = self.scaler.transform(input_array)

        # Convert to tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)

        return {
            'predicted_class': predicted_class.item(),
            'class_name': self.target_names[predicted_class.item()],
            'probabilities': probabilities.numpy().flatten(),
            'raw_output': output.numpy().flatten()
        }

    def predict_batch(self, input_values_list):
        """Predict on multiple inputs"""
        results = []
        for values in input_values_list:
            results.append(self.predict_single(values))
        return results

    def interactive_predict(self):
        """Interactive prediction loop"""

        print("\n" + "=" * 70)
        print("🔮 INTERACTIVE PREDICTION MODE")
        print("=" * 70)
        print(f"\n📋 Features ({len(self.feature_names)}):")
        for i, name in enumerate(self.feature_names):
            print(f"   {i + 1}. {name}")

        print(f"\n🎯 Target classes: {self.target_names}")
        print("\n" + "-" * 70)

        while True:
            print("\n📝 Enter feature values (comma-separated):")
            print("   Example: 12.5, 8.2, 3.1, ...")

            user_input = input("   > ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting prediction mode...")
                break

            try:
                # Parse input
                values = [float(x.strip()) for x in user_input.split(',')]

                if len(values) != len(self.feature_names):
                    print(f"❌ Expected {len(self.feature_names)} values, but got {len(values)}")
                    continue

                # Make prediction
                result = self.predict_single(values)

                # Display results
                print("\n" + "-" * 70)
                print("🎯 PREDICTION RESULT:")
                print(f"   Predicted Class: {result['predicted_class']} ({result['class_name']})")
                print(f"\n   📊 Probabilities:")

                for i, (name, prob) in enumerate(zip(self.target_names, result['probabilities'])):
                    bar = '█' * int(prob * 50)
                    print(f"      {name:12s}: {prob * 100:6.2f}% {bar}")

                print("-" * 70)

            except ValueError as e:
                print(f"❌ Error: Please enter valid numbers. {e}")
            except Exception as e:
                print(f"❌ Error: {e}")


# ============================================
# 5. MODEL PERSISTENCE
# ============================================

class ModelPersistence:
    """Save and load models with metadata"""

    @staticmethod
    def save_model(model, scaler, feature_names, target_names,
                   path='model.pth', metadata_path='model_metadata.json'):
        """Save model and metadata"""

        # Save model
        torch.save(model.state_dict(), path)
        print(f"✅ Model saved to {path}")

        # Save metadata
        metadata = {
            'feature_names': feature_names,
            'target_names': target_names,
            'n_features': len(feature_names),
            'n_classes': len(target_names),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to {metadata_path}")

    @staticmethod
    def load_model(model_class, path='model.pth', metadata_path='model_metadata.json'):
        """Load model and metadata"""

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = model_class(
            input_size=metadata['n_features'],
            output_size=metadata['n_classes']
        )

        # Load weights
        model.load_state_dict(torch.load(path))
        model.eval()

        # Recreate scaler
        scaler = StandardScaler()
        scaler.mean_ = np.array(metadata['scaler_mean'])
        scaler.scale_ = np.array(metadata['scaler_scale'])

        print(f"✅ Model loaded from {path}")
        print(f"✅ Metadata loaded from {metadata_path}")

        return model, scaler, metadata['feature_names'], metadata['target_names']


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """Main execution with interactive prediction"""

    print("=" * 70)
    print("INTERACTIVE CLASSIFICATION MODEL")
    print("Train, evaluate, and make predictions")
    print("=" * 70)

    # ===== STEP 1: Choose dataset =====
    print("\n📌 Available datasets:")
    print("   1. Breast Cancer (binary classification)")
    print("   2. Iris (multi-class classification)")
    print("   3. Synthetic (binary classification)")

    choice = input("\nSelect dataset (1-3): ").strip()

    dataset_map = {
        '1': 'breast_cancer',
        '2': 'iris',
        '3': 'make_classification'
    }

    dataset_name = dataset_map.get(choice, 'breast_cancer')

    # ===== STEP 2: Load data =====
    data_manager = DataManager()
    data = data_manager.load_dataset(dataset_name, test_size=0.2)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    target_names = data['target_names']
    n_classes = data['n_classes']
    input_size = data['input_size']
    scaler = data['scaler']

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # ===== STEP 3: Create and train model =====
    model = InteractiveClassifier(
        input_size=input_size,
        hidden_sizes=[64, 32, 16],
        output_size=n_classes,
        dropout=0.2
    )

    print(f"\n🧠 Model created:")
    print(f"   Input: {input_size} features")
    print(f"   Output: {n_classes} classes")

    # Train or load
    train_choice = input("\n🔄 Train new model or load existing? (train/load): ").strip().lower()

    if train_choice == 'load':
        try:
            model, scaler, feature_names, target_names = ModelPersistence.load_model(
                InteractiveClassifier,
                'interactive_model.pth',
                'model_metadata.json'
            )
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ No saved model found. Training new model...")
            train_choice = 'train'

    if train_choice == 'train':
        trainer = ModelTrainer()
        history = trainer.train(
            model, X_train_t, y_train_t,
            X_test_t, y_test_t,
            epochs=100,
            batch_size=32,
            lr=0.001
        )

        # Evaluate
        metrics = trainer.evaluate(model, X_test_t, y_test_t, target_names)
        trainer.plot_results(history, metrics, target_names, dataset_name)

        # Save model
        print("\n💾 Saving model...")
        ModelPersistence.save_model(model, scaler, feature_names, target_names,
                                    'interactive_model.pth', 'model_metadata.json')

    # ===== STEP 4: Interactive prediction =====
    predictor = InteractivePredictor(model, scaler, feature_names, target_names)
    predictor.interactive_predict()

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Files saved:")
    print("   - interactive_model.pth (model weights)")
    print("   - model_metadata.json (model metadata)")
    print("   - model_results_*.png (visualizations)")
    print("=" * 70)


# ============================================
# 7. SEPARATE PREDICTION FUNCTION (FOR API USE)
# ============================================

def predict_from_file(input_file='input.json', model_path='interactive_model.pth'):
    """
    Make predictions from a JSON file
    Input file format: {"features": [[1.2, 3.4, 5.6, ...]]}
    """

    try:
        # Load model
        model, scaler, feature_names, target_names = ModelPersistence.load_model(
            InteractiveClassifier, model_path, 'model_metadata.json'
        )

        # Load input
        with open(input_file, 'r') as f:
            data = json.load(f)

        features_list = data['features']
        predictor = InteractivePredictor(model, scaler, feature_names, target_names)
        results = predictor.predict_batch(features_list)

        # Save results
        output = {
            'predictions': [{
                'class': r['predicted_class'],
                'class_name': r['class_name'],
                'probabilities': r['probabilities'].tolist()
            } for r in results]
        }

        with open('predictions.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("✅ Predictions saved to predictions.json")
        return results

    except FileNotFoundError:
        print("❌ Model not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    main()