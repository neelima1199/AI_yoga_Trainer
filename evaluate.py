"""
Evaluation Script for Yoga Pose CNN.
Generates detailed metrics, confusion matrix, and training curves.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on Colab and headless servers)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from config import (
    BEST_MODEL_PATH, MODEL_SAVE_PATH, RESULTS_DIR,
    TEST_DIR, IMG_SIZE, BATCH_SIZE, CLASS_NAMES
)
from data_loader import load_datasets


def plot_training_curves(history_path, save_path):
    """Plot training and validation loss/accuracy curves."""
    with open(history_path, "r") as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")
    
    epochs = range(1, len(history["loss"]) + 1)
    
    # ---- Accuracy ----
    axes[0].plot(epochs, history["accuracy"], "b-o", label="Training Accuracy", markersize=3)
    axes[0].plot(epochs, history["val_accuracy"], "r-o", label="Validation Accuracy", markersize=3)
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Add Stage separator line
    axes[0].axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="Stage 1 → Stage 2")
    
    # ---- Loss ----
    axes[1].plot(epochs, history["loss"], "b-o", label="Training Loss", markersize=3)
    axes[1].plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=3)
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[1].axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="Stage 1 → Stage 2")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix — Yoga Pose Classification", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔲 Confusion matrix saved to: {save_path}")


def plot_sample_predictions(model, test_ds, save_path, num_samples=15):
    """Show a grid of sample predictions with correct/incorrect labels."""
    images_list = []
    labels_list = []
    preds_list = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        images_list.append(images.numpy())
        labels_list.append(np.argmax(labels.numpy(), axis=1))
        preds_list.append(np.argmax(preds, axis=1))
        if len(np.concatenate(labels_list)) >= num_samples:
            break
    
    all_images = np.concatenate(images_list)[:num_samples]
    all_labels = np.concatenate(labels_list)[:num_samples]
    all_preds = np.concatenate(preds_list)[:num_samples]
    
    # Rescale images from [-1, 1] back to [0, 1] for display
    all_images = (all_images + 1.0) / 2.0
    all_images = np.clip(all_images, 0, 1)
    
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold")
    
    for idx in range(num_samples):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        
        ax.imshow(all_images[idx])
        ax.axis("off")
        
        true_label = CLASS_NAMES[all_labels[idx]]
        pred_label = CLASS_NAMES[all_preds[idx]]
        is_correct = all_labels[idx] == all_preds[idx]
        
        color = "green" if is_correct else "red"
        symbol = "✅" if is_correct else "❌"
        ax.set_title(
            f"{symbol} True: {true_label}\nPred: {pred_label}",
            fontsize=10, color=color, fontweight="bold"
        )
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Sample predictions saved to: {save_path}")


def evaluate():
    """Run full evaluation on the test set."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("📊 FULL MODEL EVALUATION")
    print("=" * 60)
    
    # ---- Load model ----
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_SAVE_PATH
    print(f"\n📂 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # ---- Load test data ----
    _, _, test_ds, _ = load_datasets()
    
    # ---- Get predictions ----
    print("\n🔮 Generating predictions on test set...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ---- Overall metrics ----
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\n🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Test Loss:     {test_loss:.4f}")
    
    # ---- Classification report ----
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(f"\n📋 Classification Report:\n{report}")
    
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n\n")
        f.write(report)
    print(f"📄 Report saved to: {report_path}")
    
    # ---- Confusion matrix ----
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # ---- Training curves (if history exists) ----
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    if os.path.exists(history_path):
        curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
        plot_training_curves(history_path, curves_path)
    
    # ---- Sample predictions ----
    samples_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
    plot_sample_predictions(model, test_ds, samples_path)
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print(f"   All results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
