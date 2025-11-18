import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

data_dir = "./dogs-cats-mini"

filepaths = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        filepath = os.path.join(data_dir, filename)
        filepaths.append(filepath)
        if filename.startswith("cat"):
            labels.append("cat")
        elif filename.startswith("dog"):
            labels.append("dog")

df = pd.DataFrame({"filename": filepaths, "label": labels})

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
)

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)


def create_transfer_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3)
    )
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = create_transfer_model()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=0.00001
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("dvc/training_hist_transfer.png")


plot_training_history(history)

test_loss, test_accuracy = model.evaluate(test_generator, steps=50)
print(f"Test Accuracy (50 steps): {test_accuracy:.4f}")
print(f"Test Loss (50 steps): {test_loss:.4f}")

y_pred = model.predict(test_generator, steps=50)
y_true_sample = test_generator.classes[: len(y_pred)]
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

cm = confusion_matrix(y_true_sample, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"],
)
plt.title("Confusion Matrix – Transfer Learning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("dvc/conf_matrix_transfer.png")

print(classification_report(y_true_sample, y_pred_classes, target_names=["Cat", "Dog"]))
