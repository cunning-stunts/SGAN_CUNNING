from __future__ import print_function, division

import os
# if __debug__:
#     print("Debugging, so running on CPU!")
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random

import win32file
win32file._setmaxstdio(2048)

import pathlib
from collections import Counter
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

from create_gif import natural_keys
from utils import get_random_string


def get_latest_epoch(param):
    keys = [natural_keys(x)[1] for x in param]
    return int(max(keys) + 1)


class SGAN:
    def __init__(self, _run_id=None):
        if _run_id is None:
            self.run_id = get_random_string(8)
        else:
            self.run_id = _run_id
        print(f"self.run_id: {self.run_id}")

        pathlib.Path(f"images/{self.run_id}").mkdir(parents=True, exist_ok=True)
        log_path = f"logs/{self.run_id}"
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/generator").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/discriminator").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/combined").mkdir(parents=True, exist_ok=True)

        self.callback = TensorBoard(log_path)
        self.epoch_offset = 0

        self.train_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            # rescale=1 / 127.5
        )

        self.train_generator = self.train_gen.flow_from_directory(
            os.path.join("C:/Users", "xfant", "PycharmProjects", "tinder", "photos"),
            # target_size=(280, 280),
            target_size=(28, 28),
            batch_size=32,
            # batch_size=2,
            class_mode='binary',
            # color_mode='grayscale'
            color_mode='rgb'
        )

        print("Loading images...")
        files = self.train_generator.filepaths
        random.seed(1337)
        shuffle(files)
        random.seed()
        sample_of_data = []
        for x in files[:2048]:
            img = load_img(x)
            img2 = img.copy()
            img.close()
            sample_of_data.append(img2)
        # sample_of_data = np.array(sample_of_data)
        print("Fitting...")
        self.train_gen.fit(sample_of_data)
        del sample_of_data

        self.img_shape = self.train_generator.image_shape
        self.num_classes = self.train_generator.num_classes
        self.latent_dim = 30_000
        # self.latent_dim = 200 # 300 is likely the limit for this GPU at batch size 2 and img [280,280,3]
        self.rows, self.columns = 3, 4
        np.random.seed(1337)
        self.img_save_noise = np.random.normal(0, 1, (self.rows * self.columns, self.latent_dim))
        np.random.seed(None)
        print(f"img_shape: {self.img_shape}")
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)
        self.callback.set_model(self.combined)
        if _run_id:
            latest_generator_path = self.get_latest_weights_path(_run_id, "generator")
            latest_discriminator_path = self.get_latest_weights_path(_run_id, "discriminator")
            self.epoch_offset = get_latest_epoch([latest_generator_path, latest_discriminator_path])
            try:
                self.generator.load_weights(latest_generator_path)
            except Exception as e:
                print(e)
            try:
                self.discriminator.load_weights(latest_discriminator_path)
            except Exception as e:
                print(e)
            try:
                self.combined.load_weights(self.get_latest_weights_path(_run_id, "combined"))
            except Exception as e:
                print(e)

    def load_latest_model(self, model_path, model):
        saved_models = [os.path.join(model_path, model, d) for d in os.listdir(os.path.join(model_path, model))]
        latest = max(saved_models, key=os.path.getmtime)
        return load_model(f"{latest}")

    def build_generator(self):
        model = Sequential()
        # model.add(Dense(128 * 70 * 70, activation="relu", input_dim=self.latent_dim))# for 280x280x3 images
        # model.add(Reshape((70, 70, 128)))
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))  # for 28x28x3 images
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.img_shape[-1], kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(
            self.train_generator.batch_size, kernel_size=3, strides=2,
            input_shape=self.img_shape, padding="same"
        ))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.summary()

        img = Input(shape=self.img_shape)

        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)

        return Model(img, [valid, label])

    def train(self, epochs, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1, 1)

        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = self.train_generator.batch_size // 2
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        cw1 = {class_id: max_val / num_images for class_id, num_images in counter.items()}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((self.train_generator.batch_size, 1))
        fake = np.zeros((self.train_generator.batch_size, 1))

        for epoch in range(self.epoch_offset, epochs + self.epoch_offset):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            x, y = self.train_generator.next()
            # x = x - 127.5
            if y.shape[0] != self.train_generator.batch_size:
                print(f"Got non batch size: {y.shape[0]}")
                continue

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.train_generator.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y, num_classes=self.num_classes + 1)
            fake_labels = to_categorical(
                np.full((self.train_generator.batch_size, 1), self.num_classes), num_classes=self.num_classes + 1
            )

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])

            self.write_log(['g_loss'], [g_loss], epoch)
            self.write_log(['d_loss', 'ukn_loss_1', 'ukn_loss_2', 'acc', 'op_acc'], d_loss, epoch)

            # Plot the progress
            print(
                f"{epoch} [D loss: {d_loss[0]}, ukn_1: {d_loss[1]}, ukn_2: {d_loss[2]}, "
                f"acc: {100 * d_loss[3]}%, "
                f"op_acc: {100 * d_loss[4]}%] [G loss: {g_loss}]"
            )

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                if epoch % 5000 == 0:
                    self.save_model(epoch)

    def sample_images(self, epoch):
        # gen_imgs = self.generator.predict(self.img_save_noise)
        # gen_imgs = 0.5 * gen_imgs + 0.5
        # if self.img_shape[-1] == 1:
        #     plt.imshow(gen_imgs[:, :, :, 0], cmap='gray')
        # else:
        #     plt.imshow(gen_imgs[:, :, :, :])
        # plt.savefig(f"images/{run_id}/hotnot_{epoch}.png")

        gen_imgs = self.generator.predict(self.img_save_noise)
        gen_imgs *= self.train_gen.std - 1e-6
        gen_imgs += self.train_gen.mean

        fig, axs = plt.subplots(self.rows, self.columns)
        cnt = 0
        for i in range(self.rows):
            for j in range(self.columns):
                if self.img_shape[-1] == 1:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray', interpolation='none')
                else:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :], interpolation='none')
                axs[i, j].axis('off')
                cnt += 1
        plt.savefig(f"images/{self.run_id}/hotnot_{epoch}.png")
        plt.clf()
        plt.close()

    def save_model(self, epoch):

        def save(model, model_name):
            file_name = f"models/{self.run_id}/{model_name}.h5.tmp"
            # model.save(f"models/{self.run_id}/{model_name}.h5")
            model.save_weights(file_name)
            os.rename(file_name, file_name.replace(".tmp", ""))

        save(self.generator, f"generator/sgan_generator_{epoch}")
        save(self.discriminator, f"discriminator/sgan_discriminator_{epoch}")
        save(self.combined, f"combined/sgan_adversarial_{epoch}")

    def get_latest_weights_path(self, _run_id, model_name):
        saved_models = [
            os.path.join("models", _run_id, model_name, d)
            for d in os.listdir(os.path.join("models", _run_id, model_name))
            if not d.endswith(".tmp")
        ]
        return max(saved_models, key=os.path.getmtime)

    def write_log(self, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, batch_no)
            self.callback.writer.flush()


if __name__ == '__main__':
    sgan = SGAN()
    # sgan = SGAN(_run_id="VZMGUSKD")
    sgan.train(epochs=20000, sample_interval=10)
