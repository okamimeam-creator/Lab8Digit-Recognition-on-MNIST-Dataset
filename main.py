import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_SIZE = 28
PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE


def load_training_data(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {path}. Download train.csv from the "
            "Kaggle Digit Recognizer competition and place it here, or pass "
            "its location with --train-csv."
        )

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"Training data is empty: {path}")
    if data.shape[1] != PIXEL_COUNT + 1:
        raise ValueError(
            f"Expected a label column and {PIXEL_COUNT} pixel columns in "
            f"{path}; found {data.shape[1]} columns."
        )

    try:
        labels = pd.to_numeric(data.iloc[:, 0], errors="raise").to_numpy()
        pixels = data.iloc[:, 1:].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Training data must contain numeric labels and pixels: {path}") from error

    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Training labels must be finite integers from 0 to 9.")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("Training labels must be integers from 0 to 9.")
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("Training pixels must be finite values from 0 to 255.")

    images = (pixels / 255.0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    return images, labels.astype(np.int64)


def load_test_data(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Test data not found: {path}")

    data = pd.read_csv(path)
    if data.shape[1] != PIXEL_COUNT:
        raise ValueError(
            f"Expected {PIXEL_COUNT} pixel columns in {path}; "
            f"found {data.shape[1]} columns."
        )
    try:
        pixels = data.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Test data must contain numeric pixels: {path}") from error

    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("Test pixels must be finite values from 0 to 255.")
    return (pixels / 255.0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a CNN on Kaggle's Digit Recognizer data.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("train.csv"),
        help="Path to Kaggle's labeled training CSV (default: ./train.csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("test.csv"),
        help="Path to Kaggle's unlabeled test CSV (default: ./test.csv)",
    )
    args = parser.parse_args(argv)

    X, y = load_training_data(args.train_csv)
    print(f"Training data: {X.shape}; labels: {y.shape}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, models

    print("TensorFlow version:", tf.__version__)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    plt.figure(figsize=(10, 4))
    for i in range(min(10, len(X_train))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.show()

    val_predictions = np.argmax(model.predict(X_val), axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_val, val_predictions))
    sns.heatmap(confusion_matrix(y_val, val_predictions), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    if args.test_csv.is_file():
        X_test = load_test_data(args.test_csv)
        if len(X_test):
            sample_index = min(5, len(X_test) - 1)
            prediction = int(np.argmax(model.predict(X_test[sample_index : sample_index + 1], verbose=0)))
            plt.imshow(X_test[sample_index].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
            plt.title(f"Predicted label: {prediction}")
            plt.axis("off")
            plt.show()
            print(f"Prediction for test row {sample_index}: {prediction}")
        else:
            print(f"Test data is empty; skipped prediction: {args.test_csv}")
    else:
        print(f"Test data not found; skipped test prediction: {args.test_csv}")


if __name__ == "__main__":
    main()
