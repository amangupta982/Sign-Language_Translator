"""
SignSpeak — Evaluation & Metrics Visualization

Generates confusion matrix, classification report, training curves,
and per-class accuracy charts. Saves all plots to static/plots/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from model import NUM_CLASSES, RAW_TO_INDEX, LETTERS

MODEL_PATH = "sign_model_v2.keras"
HISTORY_PATH = "training_history.json"
PLOTS_DIR = "static/plots"


def load_test_data(csv_path: str = "sign_mnist_test.csv"):
    """Load test data with correct label mapping."""
    df = pd.read_csv(csv_path)
    labels_raw = df['label'].values
    pixels = df.drop('label', axis=1).values.astype('float32') / 255.0
    images = pixels.reshape(-1, 28, 28, 1)
    labels = np.array([RAW_TO_INDEX[r] for r in labels_raw])
    return images, labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt='.1%', cmap='YlGnBu',
        xticklabels=LETTERS, yticklabels=LETTERS,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Proportion'},
        ax=ax,
    )
    ax.set_xlabel('Predicted Letter', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Letter', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {save_path}")


def plot_training_history(history_path, save_path):
    """Plot training & validation accuracy/loss curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, history['accuracy'], 'o-', color='#10B981', label='Train', markersize=3)
    ax1.plot(epochs, history['val_accuracy'], 'o-', color='#6366F1', label='Validation', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history['loss'], 'o-', color='#F59E0B', label='Train', markersize=3)
    ax2.plot(epochs, history['val_loss'], 'o-', color='#EF4444', label='Validation', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Training curves → {save_path}")


def plot_per_class_accuracy(y_true, y_pred, save_path):
    """Bar chart of per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#10B981' if acc >= 0.9 else '#F59E0B' if acc >= 0.7 else '#EF4444'
              for acc in per_class]
    bars = ax.bar(LETTERS, per_class, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Letter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=15, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color='#10B981', linestyle='--', alpha=0.4, label='90% threshold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Per-class accuracy → {save_path}")


def plot_sample_predictions(model, X_test, y_true, y_pred, save_path, n=24):
    """Grid of sample images with real vs predicted labels."""
    # Pick a mix of correct and incorrect predictions
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    n_wrong = min(8, wrong_mask.sum())
    n_correct = n - n_wrong

    correct_indices = np.where(correct_mask)[0]
    wrong_indices = np.where(wrong_mask)[0]

    np.random.seed(42)
    selected = np.concatenate([
        np.random.choice(wrong_indices, n_wrong, replace=False) if n_wrong > 0 else [],
        np.random.choice(correct_indices, n_correct, replace=False),
    ]).astype(int)
    np.random.shuffle(selected)

    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.8))
    axes = axes.flatten()

    for i, idx in enumerate(selected):
        ax = axes[i]
        img = X_test[idx].reshape(28, 28)
        true_letter = LETTERS[y_true[idx]]
        pred_letter = LETTERS[y_pred[idx]]
        is_correct = true_letter == pred_letter

        ax.imshow(img, cmap='gray')
        ax.set_title(
            f'True: {true_letter}  Pred: {pred_letter}',
            fontsize=9, fontweight='bold',
            color='#10B981' if is_correct else '#EF4444',
        )
        ax.axis('off')

    for i in range(len(selected), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green = Correct, Red = Wrong)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Sample predictions → {save_path}")


def save_classification_report(y_true, y_pred, save_path):
    """Save classification report as JSON for the web UI."""
    report = classification_report(
        y_true, y_pred, target_names=LETTERS, output_dict=True
    )
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Classification report → {save_path}")

    # Also print it
    print("\n" + classification_report(y_true, y_pred, target_names=LETTERS))


def main():
    print("=" * 60)
    print("SignSpeak — Evaluation & Metrics")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load model and test data
    print("\n[1/3] Loading model and test data...")
    model = load_model(MODEL_PATH)
    X_test, y_test = load_test_data()
    print(f"  Test samples: {len(y_test):,}")

    # Predict
    print("\n[2/3] Running predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"  Overall Accuracy: {overall_acc:.4f}")

    # Generate all plots
    print("\n[3/3] Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, f"{PLOTS_DIR}/confusion_matrix.png")
    plot_per_class_accuracy(y_test, y_pred, f"{PLOTS_DIR}/per_class_accuracy.png")
    plot_sample_predictions(model, X_test, y_test, y_pred, f"{PLOTS_DIR}/sample_predictions.png")
    save_classification_report(y_test, y_pred, f"{PLOTS_DIR}/classification_report.json")

    if os.path.exists(HISTORY_PATH):
        plot_training_history(HISTORY_PATH, f"{PLOTS_DIR}/training_history.png")

    print(f"\n{'=' * 60}")
    print(f"  All plots saved to {PLOTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
