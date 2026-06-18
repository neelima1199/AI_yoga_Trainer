"""
Data Loader for Yoga Pose Classification.
Handles dataset loading, augmentation, validation split, and class weight computation.
Designed to work with both local and Google Colab environments.
"""
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, CLASS_NAMES
)


def create_data_augmentation():
    """Create a data augmentation pipeline for training images."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),        # ±15°
        tf.keras.layers.RandomZoom((-0.2, 0.0)),      # Up to 20% zoom in
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="data_augmentation")


def load_datasets():
    """
    Load training (with validation split) and test datasets.
    
    Returns:
        train_ds: Training dataset (augmented, preprocessed)
        val_ds: Validation dataset (preprocessed, no augmentation)
        test_ds: Test dataset (preprocessed, no augmentation)
        class_weights: Dictionary mapping class index to weight
    """
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    
    # ---- Load raw datasets ----
    raw_train = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        shuffle=True,
    )
    
    raw_val = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        shuffle=False,
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=False,
    )
    
    # ---- Print dataset info ----
    train_count = raw_train.cardinality().numpy() * BATCH_SIZE
    val_count = raw_val.cardinality().numpy() * BATCH_SIZE
    test_count = test_ds.cardinality().numpy() * BATCH_SIZE
    print(f"\n📦 Training samples:   ~{train_count}")
    print(f"📦 Validation samples: ~{val_count}")
    print(f"📦 Test samples:       ~{test_count}")
    print(f"📦 Classes: {CLASS_NAMES}")
    
    # ---- Compute class weights ----
    class_weights = _compute_class_weights()
    print(f"\n⚖️  Class weights: {class_weights}")
    
    # ---- Data augmentation (training only) ----
    data_augmentation = create_data_augmentation()
    
    # Apply augmentation + MobileNetV2 preprocessing to training set
    train_ds = raw_train.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    # Only preprocessing (no augmentation) for val and test
    val_ds = raw_val.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    test_ds = test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    print("\n✅ Datasets loaded and preprocessed successfully!")
    print("=" * 60)
    
    return train_ds, val_ds, test_ds, class_weights


def _compute_class_weights():
    """
    Compute class weights based on the number of images per class 
    in the training directory to handle class imbalance.
    """
    class_counts = []
    labels = []
    
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            count = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            class_counts.append(count)
            labels.extend([idx] * count)
    
    print(f"\n📊 Training class distribution:")
    for name, count in zip(CLASS_NAMES, class_counts):
        print(f"   {name:12s}: {count:4d} images")
    
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    return dict(enumerate(weights))


if __name__ == "__main__":
    # Quick test
    train_ds, val_ds, test_ds, class_weights = load_datasets()
    
    # Show a sample batch
    for images, labels in train_ds.take(1):
        print(f"\nSample batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Pixel range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        break
