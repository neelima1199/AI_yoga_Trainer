"""
CNN Model for Yoga Pose Classification.
Uses MobileNetV2 as the backbone with a custom classification head.
Supports two-stage training: feature extraction → fine-tuning.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

from config import (
    IMG_SHAPE, NUM_CLASSES, DENSE_UNITS,
    DROPOUT_1, DROPOUT_2, L2_REG, FINE_TUNE_AT
)


def build_model(for_training=True):
    """
    Build the yoga pose classification model using MobileNetV2 transfer learning.
    
    Args:
        for_training: If True, backbone is frozen for Stage 1 training.
                      If False, loads for inference only.
    
    Returns:
        tf.keras.Model
    """
    print("\n🔨 Building model...")
    
    # ---- Base model: MobileNetV2 pretrained on ImageNet ----
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,          # Remove the original 1000-class classifier
        weights="imagenet"
    )
    
    # Freeze the backbone for Stage 1 (feature extraction)
    if for_training:
        base_model.trainable = False
    
    # ---- Custom classification head ----
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    
    # Pass through base model
    x = base_model(inputs, training=False)  # training=False keeps BatchNorm in inference mode
    
    # Global average pooling: (batch, 7, 7, 1280) → (batch, 1280)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Regularized dense layers
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_2)(x)
    
    # Output layer: 5 yoga pose classes
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="YogaPoseCNN")
    
    # ---- Print model summary ----
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    
    print(f"   📐 Total parameters:       {total_params:,}")
    print(f"   🏋️  Trainable parameters:   {trainable_params:,}")
    print(f"   🔒 Non-trainable params:   {non_trainable_params:,}")
    print(f"   🔢 Output classes:          {NUM_CLASSES}")
    print(f"   📏 Input shape:             {IMG_SHAPE}")
    
    return model


def unfreeze_for_fine_tuning(model):
    """
    Unfreeze the last portion of the MobileNetV2 backbone for Stage 2 fine-tuning.
    
    Args:
        model: The compiled keras model.
    
    Returns:
        model with partially unfrozen backbone
    """
    print("\n🔓 Unfreezing backbone for fine-tuning...")
    
    # Get the base model (first layer in our Model)
    base_model = model.layers[1]  # MobileNetV2 is the second layer (after Input)
    base_model.trainable = True
    
    total_layers = len(base_model.layers)
    
    # Freeze everything before FINE_TUNE_AT, unfreeze the rest
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    unfrozen_count = total_layers - FINE_TUNE_AT
    trainable_params = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    
    print(f"   📊 Total backbone layers: {total_layers}")
    print(f"   🔒 Frozen layers:         {FINE_TUNE_AT}")
    print(f"   🔓 Unfrozen layers:       {unfrozen_count}")
    print(f"   🏋️  Trainable parameters:  {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Quick test: build and display model
    model = build_model(for_training=True)
    model.summary()
    
    print("\n--- After fine-tuning setup ---")
    model = unfreeze_for_fine_tuning(model)
