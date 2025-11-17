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

data_dir = "./dogs-cats-mini"

filepaths = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        filepath = os.path.join(data_dir, filename)
        filepaths.append(filepath)

        # Wyciąganie etykiety z nazwy pliku
        if filename.startswith("cat"):
            labels.append("cat")
        elif filename.startswith("dog"):
            labels.append("dog")

# Tworzenie DataFrame
df = pd.DataFrame({"filename": filepaths, "label": labels})

print(f"Łączna liczba obrazów: {len(df)}")
print(df["label"].value_counts())

# Podział na zbiór treningowy i testowy
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
)

print(f"Zbiór treningowy: {len(train_df)}")
print(f"Zbiór walidacyjny: {len(val_df)}")
print(f"Zbiór testowy: {len(test_df)}")

# Parametry
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50

# Augmentacja danych dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest",
)

# Tylko normalizacja dla zbiorów walidacyjnego i testowego
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generatory danych
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


# Budowa modelu CNN
def create_model_v1():
    model = keras.Sequential(
        [
            # Warstwy konwolucyjne
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            # Warstwy gęste
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Tworzenie i trening modelu
model_v1 = create_model_v1()
model_v1.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001)

# Trening modelu
history_v1 = model_v1.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# Funkcje do wizualizacji wyników
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Krzywa dokładności
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Krzywa straty
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("dvc/training_hist.png")


# Wizualizacja krzywej uczenia
plot_training_history(history_v1)


best_model = model_v1

# Ocena najlepszego modelu na zbiorze testowym (też z ograniczonymi krokami dla szybkości)
test_loss, test_accuracy = best_model.evaluate(test_generator, steps=50)
print(f"Test Accuracy (50 steps): {test_accuracy:.4f}")
print(f"Test Loss (50 steps): {test_loss:.4f}")

# Szybka macierz pomyłek (na mniejszej próbce)
y_pred_best = best_model.predict(test_generator, steps=50)
y_true_sample = test_generator.classes[: len(y_pred_best)]
y_pred_classes_best = (y_pred_best > 0.5).astype(int).flatten()

cm_best = confusion_matrix(y_true_sample, y_pred_classes_best)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_best,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"],
)
plt.title(f"Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("dvc/conf_matrix.png")

print(
    classification_report(
        y_true_sample, y_pred_classes_best, target_names=["Cat", "Dog"]
    )
)
