# %% STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)

# %% STEP 2: Load CSV file
# Update path as needed or keep 'train.csv' if placed in the same directory
import os
csv_path = 'train.csv' if os.path.exists('train.csv') else r"C:/Users/dell pc/Downloads/digit-recognizer/train.csv"
df = pd.read_csv(csv_path)
print("Data shape:", df.shape)
df.head()

# %% STEP 3: Data Preprocessing
# Label save as y and pixels save as X
y = df.iloc[:, 0].values            
X = df.iloc[:, 1:].values           

# Normalization: pixel values 0-255 to 0-1 range
X = X.astype("float32") / 255.0

# Reshape: 784 -> 28 x 28 x 1 (height, width, channel)
X = X.reshape(-1, 28, 28, 1)

print("X shape:", X.shape, " y shape:", y.shape)

# %% STEP 4: Training and Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Validation:", X_val.shape)

# %% STEP 5: Show some images (data check)
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% STEP 6: CNN model building
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # --- Block 1 ---
    layers.Conv2D(32, (3, 3)),          # Convolution (32 filters, 3x3)
    layers.Activation("relu"),          # ReLU
    layers.MaxPooling2D((2, 2)),        # Max Pooling 2x2

    # --- Block 2 ---
    layers.Conv2D(64, (3, 3)),          # Convolution (64 filters)
    layers.Activation("relu"),          # ReLU
    layers.MaxPooling2D((2, 2)),        # Max Pooling

    # --- Classifier ---
    layers.Flatten(),                   # 2D feature maps -> 1D vector
    layers.Dense(128),                  # Fully Connected layer
    layers.Activation("relu"),
    layers.Dropout(0.3),                # Reduce overfitting
    layers.Dense(10),                   # 10 classes (digits 0-9)
    layers.Activation("softmax"),       # Softmax -> probabilities
])

model.summary()

# %% STEP 7: Model compile 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # labels integer (0-9)
    metrics=["accuracy"],
)

# %% STEP 8: Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
)

# %% STEP 9: Plot Training Accuracy and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
plt.show()

# %% STEP 10: Model Evaluation & Confusion Matrix
val_preds = model.predict(X_val)
val_pred_labels = np.argmax(val_preds, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_val, val_pred_labels))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, val_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% STEP 11: Prediction on Test Dataset / Custom Image
test_csv_path = 'test.csv' if os.path.exists('test.csv') else r"C:/Users/dell pc/Downloads/digit-recognizer/test.csv"
if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    X_test = (test_df.values.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

    idx = 5                                    # sample index
    probs = model.predict(X_test[idx:idx+1], verbose=0)
    label = np.argmax(probs)

    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted label: {label}")
    plt.axis("off")
    plt.show()
    print("Predicted label:", label)
else:
    print("test.csv path not found. Please place test.csv in the working directory.")