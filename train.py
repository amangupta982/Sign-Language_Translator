"""
SignSpeak — Training Pipeline

Trains the CNN on Sign MNIST with data augmentation, early stopping,
learning rate scheduling, and proper label mapping.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from model import build_model, NUM_CLASSES, RAW_TO_INDEX

MODEL_PATH = "sign_model_v2.keras"
HISTORY_PATH = "training_history.json"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3


def load_data(csv_path: str):
    """Load Sign MNIST CSV and return images (N,28,28,1) and index labels."""
    df = pd.read_csv(csv_path)
    labels_raw = df['label'].values
    pixels = df.drop('label', axis=1).values.astype('float32')

    # Normalize pixels to [0, 1]
    pixels = pixels / 255.0

    # Reshape to (N, 28, 28, 1)
    images = pixels.reshape(-1, 28, 28, 1)

    # Map raw labels (0-24, skip 9) → contiguous indices (0-23)
    labels = np.array([RAW_TO_INDEX[r] for r in labels_raw])

    return images, labels


def main():
    print("=" * 60)
    print("SignSpeak — Training Pipeline")
    print("=" * 60)

    # --- Load Data ---
    print("\n[1/5] Loading datasets...")
    X_train, y_train = load_data("sign_mnist_train.csv")
    X_test, y_test = load_data("sign_mnist_test.csv")

    # Split training into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"  Train:      {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test:       {X_test.shape[0]:,} samples")
    print(f"  Classes:    {NUM_CLASSES}")

    # One-hot encode
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    # --- Data Augmentation ---
    print("\n[2/5] Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        horizontal_flip=False,  # signs are NOT left-right symmetric
    )
    datagen.fit(X_train)

    # --- Build Model ---
    print("\n[3/5] Building CNN model...")
    model = build_model(NUM_CLASSES)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --- Train ---
    print("\n[4/5] Training...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
    )

    # --- Save history ---
    print("\n[5/5] Saving artifacts...")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # Final evaluation on test set
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
    print(f"{'=' * 60}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print("\nNext step: run  python evaluate.py  to generate metrics plots.")


if __name__ == "__main__":
    main()
