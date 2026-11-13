import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load Dataset (Multiclass - Digits Dataset)
print("Loading digits dataset...")
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Convert to PyTorch Tensors
print("\nConverting to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"X_train tensor shape: {X_train_tensor.shape}")
print(f"y_train tensor shape: {y_train_tensor.shape}")
print(f"X_train tensor dtype: {X_train_tensor.dtype}")
print(f"y_train tensor dtype: {y_train_tensor.dtype}")

# 5. Create DataLoader for batch processing
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 6. Define Neural Network Model
class MulticlassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulticlassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# 7. Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(y))

model = MulticlassNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel architecture:")
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Number of classes: {num_classes}")


# 8. Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, train_accuracies


print("\nTraining the model...")
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, epochs=50)


# 9. Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_predictions.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())

    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return all_predictions, all_labels, accuracy


test_predictions, test_labels, test_accuracy = evaluate_model(model, test_loader)

# 10. Visualization with Matplotlib
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Training Loss
axes[0, 0].plot(train_losses, 'b-', linewidth=2)
axes[0, 0].set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training Accuracy
axes[0, 1].plot(train_accuracies, 'g-', linewidth=2)
axes[0, 1].set_title('Training Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sample digits from the dataset
sample_indices = np.random.choice(range(len(X)), 10, replace=False)
for i, idx in enumerate(sample_indices[:5]):
    axes[0, 2].imshow(X[idx].reshape(8, 8), cmap='gray')
    axes[0, 2].set_title(f'Digit: {y[idx]}', fontsize=10)
    axes[0, 2].axis('off')

# Plot 4: Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Plot 5: Class Distribution
class_counts = np.bincount(y)
axes[1, 1].bar(range(len(class_counts)), class_counts, color='skyblue', edgecolor='black')
axes[1, 1].set_title('Class Distribution in Dataset', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Digit Class')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xticks(range(len(class_counts)))

# Plot 6: Feature Distribution (First 10 features)
features_to_plot = 10
feature_means = X_train_scaled[:, :features_to_plot].mean(axis=0)
feature_stds = X_train_scaled[:, :features_to_plot].std(axis=0)

axes[1, 2].bar(range(features_to_plot), feature_means, yerr=feature_stds,
               capsize=5, color='lightcoral', edgecolor='black')
axes[1, 2].set_title('Feature Distribution (First 10 Features)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Feature Index')
axes[1, 2].set_ylabel('Mean Value')

plt.tight_layout()
plt.savefig('multiclass_classification_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Print classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions,
                            target_names=[f'Digit {i}' for i in range(10)]))

# 12. Additional Tensor Operations Demonstration
print("\n" + "=" * 50)
print("Tensor Operations Demonstration")
print("=" * 50)

# Demonstrate tensor conversion from different formats
print("\n1. Converting numpy array to tensor:")
numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"NumPy array: {numpy_array}")
print(f"Tensor: {tensor_from_numpy}")

print("\n2. Tensor shape manipulation:")
sample_tensor = X_train_tensor[:5]
print(f"Original shape: {sample_tensor.shape}")
print(f"Reshaped to 2D: {sample_tensor.reshape(5, -1).shape}")
print(f"Reshaped to 1D: {sample_tensor.flatten().shape}")

print("\n3. Tensor operations:")
tensor_a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
tensor_b = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)
print(f"Tensor A: {tensor_a}")
print(f"Tensor B: {tensor_b}")
print(f"A + B: {tensor_a + tensor_b}")
print(f"A * B: {tensor_a * tensor_b}")
print(f"Mean of A: {tensor_a.mean().item():.2f}")

print("\n4. GPU Check (if available):")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Moving tensor to GPU: {tensor_a.to(device)}")
else:
    print("GPU not available, using CPU")

print("\n5. Saving and loading tensors:")
torch.save(X_train_tensor, 'tensor_data.pt')
loaded_tensor = torch.load('tensor_data.pt')
print(f"Original tensor shape: {X_train_tensor.shape}")
print(f"Loaded tensor shape: {loaded_tensor.shape}")
print(f"Are they equal? {torch.allclose(X_train_tensor, loaded_tensor)}")

print("\n" + "=" * 50)
print("Classification complete! Check the visualization plot for results.")
print("=" * 50)