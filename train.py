"""
Training Pipeline for Yoga Pose CNN.
Two-stage training: Feature Extraction → Fine-Tuning.
Designed to run on Google Colab with GPU acceleration.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from config import (
    MODELS_DIR, RESULTS_DIR, MODEL_SAVE_PATH, BEST_MODEL_PATH,
    STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE1_LR, STAGE2_LR,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE
)
from data_loader import load_datasets
from model import build_model, unfreeze_for_fine_tuning


def get_callbacks(stage_name):
    """Create training callbacks for a given stage."""
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        # Stop training if no improvement
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    return callbacks


def train():
    """Execute the full two-stage training pipeline."""
    
    # ---- Create output directories ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ---- Check GPU availability ----
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU detected: {gpus[0].name}")
        print(f"   Using GPU for training — this will be fast!")
    else:
        print("⚠️  No GPU detected — training on CPU (will be slower)")
    
    # ---- Load data ----
    train_ds, val_ds, test_ds, class_weights = load_datasets()

    # ✅ FIX: Skip corrupted/broken images
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    # ---- Build model ----
    model = build_model(for_training=True)
    
    # ==================================================================
    # STAGE 1: Feature Extraction (Frozen Backbone)
    # ==================================================================
    print("\n" + "=" * 60)
    print("🏋️  STAGE 1: FEATURE EXTRACTION")
    print("   Backbone: FROZEN (only training classifier head)")
    print(f"   Epochs:   {STAGE1_EPOCHS}")
    print(f"   LR:       {STAGE1_LR}")
    print("=" * 60 + "\n")
    
    model.compile(
        optimizer=Adam(learning_rate=STAGE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("stage1"),
        verbose=1
    )
    
    # Print Stage 1 results
    best_val_acc_s1 = max(history1.history["val_accuracy"])
    print(f"\n📊 Stage 1 Best Validation Accuracy: {best_val_acc_s1:.4f}")
    
    # ==================================================================
    # STAGE 2: Fine-Tuning (Partially Unfrozen Backbone)
    # ==================================================================
    print("\n" + "=" * 60)
    print("🏋️  STAGE 2: FINE-TUNING")
    print("   Backbone: PARTIALLY UNFROZEN (last ~30%)")
    print(f"   Epochs:   {STAGE2_EPOCHS}")
    print(f"   LR:       {STAGE2_LR}")
    print("=" * 60 + "\n")
    
    # Unfreeze the last portion of the backbone
    model = unfreeze_for_fine_tuning(model)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=STAGE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("stage2"),
        verbose=1
    )
    
    # Print Stage 2 results
    best_val_acc_s2 = max(history2.history["val_accuracy"])
    print(f"\n📊 Stage 2 Best Validation Accuracy: {best_val_acc_s2:.4f}")
    
    # ==================================================================
    # SAVE FINAL MODEL & TRAINING HISTORY
    # ==================================================================
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Final model saved to: {MODEL_SAVE_PATH}")
    
    # Merge training histories
    full_history = {}
    for key in history1.history:
        full_history[key] = history1.history[key] + history2.history[key]
    
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        # Convert numpy types to native Python types for JSON
        serializable = {k: [float(v) for v in vals] for k, vals in full_history.items()}
        json.dump(serializable, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")
    
    # ==================================================================
    # QUICK TEST EVALUATION
    # ==================================================================
    print("\n" + "=" * 60)
    print("🧪 QUICK TEST SET EVALUATION")
    print("=" * 60)
    
    # Load the best model for evaluation
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    
    print(f"\n🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📉 Test Loss:     {test_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best validation accuracy: {max(best_val_acc_s1, best_val_acc_s2):.4f}")
    print(f"   Test accuracy:            {test_accuracy:.4f}")
    print(f"   Model saved to:           {MODEL_SAVE_PATH}")
    print(f"   Best model saved to:      {BEST_MODEL_PATH}")
    print("=" * 60)
    
    return model, full_history


if __name__ == "__main__":
    model, history = train()
