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

# Parametry - ZREDUKOWANE dla szybszego treningu
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5  # Zmniejszone z 30 do 10
STEPS_PER_EPOCH = 100  # Zmniejszone z ~500 do 100
VALIDATION_STEPS = 50  # Zmniejszone dla walidacji

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


# Budowa modelu CNN - uproszczona dla szybszego treningu
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
            # Warstwy gęste
            layers.Flatten(),
            layers.Dense(256, activation="relu"),  # Zmniejszone z 512
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Tworzenie i trening modelu
model_v1 = create_model_v1()
model_v1.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)  # Zmniejszone patience
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001
)  # Zmniejszone patience

print("Rozpoczynanie treningu Model V1...")
# Trening modelu ze zmniejszoną liczbą kroków
history_v1 = model_v1.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,  # Użycie zmniejszonej liczby kroków
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,  # Użycie zmniejszonej liczby kroków walidacji
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
    plt.savefig("training_hist.png")


# Wizualizacja krzywej uczenia
plot_training_history(history_v1)

# SZYBSZE MODELE DO PORÓWNANIA


# Model V2 - Zoptymalizowany dla szybkiego treningu
def create_model_v2():
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),  # Szybsze niż Flatten + Dense
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


print("Rozpoczynanie treningu Model V2...")
model_v2 = create_model_v2()
history_v2 = model_v2.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# Model V3 - Bardziej zwarta architektura
def create_model_v3():
    model = keras.Sequential(
        [
            layers.Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


print("Rozpoczynanie treningu Model V3...")
model_v3 = create_model_v3()
history_v3 = model_v3.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# SZYBKIE PORÓWNANIE MODELI
def compare_models(histories, model_names):
    plt.figure(figsize=(15, 5))

    # Porównanie dokładności
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history["val_accuracy"], label=name)
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Porównanie straty
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history.history["val_loss"], label=name)
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("acc-loss-comp.png")


# Ocena wszystkich modeli
models = [model_v1, model_v2, model_v3]
model_names = ["Model V1 (Basic)", "Model V2 (Optimized)", "Model V3 (Compact)"]
histories = [history_v1, history_v2, history_v3]

compare_models(histories, model_names)

# SZYBKA OCENA NAJLEPSZEGO MODELU
best_model_idx = np.argmax([max(h.history["val_accuracy"]) for h in histories])
best_model = models[best_model_idx]
best_model_name = model_names[best_model_idx]

print(f"NAJLEPSZY MODEL: {best_model_name}")

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
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("conf_matrix.png")

print(
    classification_report(
        y_true_sample, y_pred_classes_best, target_names=["Cat", "Dog"]
    )
)

print("\nTRENING ZAKOŃCZONY! Czas treningu znacząco skrócony:")
print(f"- Epoki: {EPOCHS} (zamiast 30)")
print(f"- Kroki na epokę: {STEPS_PER_EPOCH} (zamiast ~500)")
print(f"- Łącznie: {EPOCHS * STEPS_PER_EPOCH} kroków treningowych")
