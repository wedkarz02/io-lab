import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 50

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)


def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, input_dim=LATENT_DIM))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.Conv2D(1, 3, padding="same", activation="sigmoid"))
    return model


def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, 3, strides=2, padding="same", input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, 3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy"
)

discriminator.trainable = False

gan_input = keras.Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")


def save_images(epoch, generator):
    noise = np.random.normal(0, 1, (25, LATENT_DIM))
    gen_imgs = generator.predict(noise)
    gen_imgs = gen_imgs * 255.0
    gen_imgs = gen_imgs.reshape(25, 28, 28)
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    idx = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[idx], cmap="gray")
            axs[i, j].axis("off")
            idx += 1
    fig.savefig(f"gan_img_new/mnist_gan_epoch_{epoch}.png")
    plt.close()


for epoch in range(EPOCHS):
    for real_imgs in dataset:
        batch_size = real_imgs.shape[0]
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_imgs = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        gan_labels = np.ones((batch_size, 1))

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, gan_labels)

    print(f"Epoch {epoch+1}/{EPOCHS} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        save_images(epoch + 1, generator)
