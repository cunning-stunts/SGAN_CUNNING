from __future__ import print_function, division

import os
import pathlib
# if __debug__:
#     print("Debugging, so running on CPU!")
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import Counter
from random import shuffle

import keras.backend as K
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

# for rtx 20xx cards
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_latest_step(param):
    keys = [natural_keys(x)[1] for x in param]
    return int(max(keys) + 1)


class SGAN:
    def __init__(self, _run_id=None):

        # target_size = (28, 28)
        # target_size = (56, 56)
        target_size = (112, 112)
        self.channels = 1
        self.latent_dim = 100
        self.batch_size = 64
        self.generator_feature_amount = 128
        self.amount_of_generator_layer_units = 2
        self.min_generator_feature_size = int(64)
        # self.min_generator_feature_size = int(2.0 * target_size[0])
        # self.learning_rate = 0.00001
        self.learning_rate = 0.0002
        self.adam_beta1 = 0.99
        # self.adam_beta1 = 0.5
        self.clip_value = 0.01

        if _run_id is None:
            self.run_id = get_random_string(8)
        else:
            self.run_id = _run_id
        np.random.seed(1337)
        random.seed(1337)
        print(f"self.run_id: {self.run_id}")

        pathlib.Path(f"images/{self.run_id}").mkdir(parents=True, exist_ok=True)
        log_path = f"logs/{self.run_id}"
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/generator").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/discriminator").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"models/{self.run_id}/combined").mkdir(parents=True, exist_ok=True)

        self.callback = TensorBoard(log_path)
        self.step_offset = 0

        self.train_gen = ImageDataGenerator(
            rescale=1 / 127.5
        )

        if self.channels == 1:
            self.color_mode = 'grayscale'
        elif self.channels == 3:
            self.color_mode = "rgb"
        else:
            raise

        self.train_generator = self.train_gen.flow_from_directory(
            os.path.join("C:/Users", "xfant", "PycharmProjects", "sgan", "mnist"),
            # os.path.join("C:/Users", "xfant", "PycharmProjects", "tinder", "photos"),
            target_size=target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode=self.color_mode
        )

        self.img_shape = self.train_generator.image_shape
        self.num_classes = self.train_generator.num_classes
        self.rows, self.columns = 3, 4
        self.img_save_noise = np.random.normal(0, 1, (self.rows * self.columns, self.latent_dim))
        print(f"img_shape: {self.img_shape}")
        optimizer = Adam(self.learning_rate, self.adam_beta1)
        # optimizer = Adam()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            # loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss=[self.wasserstein_loss, 'categorical_crossentropy'],
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
        self.combined.compile(
            loss=self.wasserstein_loss, optimizer=optimizer
        )
        self.callback.set_model(self.combined)
        if _run_id:
            latest_generator_path = self.get_latest_weights_path(_run_id, "generator")
            latest_discriminator_path = self.get_latest_weights_path(_run_id, "discriminator")
            self.step_offset = get_latest_step([latest_generator_path, latest_discriminator_path])
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

        half_batch = self.train_generator.batch_size // 2
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        self.cw1 = {class_id: max_val / num_images for class_id, num_images in counter.items()}
        self.cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        self.cw2[self.num_classes] = 1 / half_batch

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def populate_std_mean(self, target_size):
        # doesn't seem to work!
        print("Loading images...")
        files = self.train_generator.filepaths
        shuffle(files)
        sample_of_data = [
            np.array(load_img(img, target_size=target_size, color_mode=self.color_mode).copy()) for img in files
        ]
        sample_of_data = np.array(sample_of_data)
        sample_of_data = np.expand_dims(sample_of_data, axis=-1)
        print("Fitting...")
        self.train_gen.fit(sample_of_data)
        del sample_of_data
        print(f"Mean: {self.train_gen.mean}")
        print(f"std: {self.train_gen.std}")

    def load_latest_model(self, model_path, model):
        saved_models = [os.path.join(model_path, model, d) for d in os.listdir(os.path.join(model_path, model))]
        latest = max(saved_models, key=os.path.getmtime)
        return load_model(f"{latest}")

    def build_generator(self):
        model = Sequential()
        x_units = int(self.img_shape[0] * (0.5 ** self.amount_of_generator_layer_units))
        y_units = int(self.img_shape[1] * (0.5 ** self.amount_of_generator_layer_units))

        model.add(Dense(
            self.generator_feature_amount * x_units * y_units,
            activation="relu", input_dim=self.latent_dim)
        )
        model.add(Reshape((x_units, y_units, self.generator_feature_amount)))
        layer_features = self.generator_feature_amount
        model.add(BatchNormalization(momentum=0.8))
        for _ in range(self.amount_of_generator_layer_units):
            model.add(UpSampling2D())
            model.add(Conv2D(layer_features, kernel_size=3, padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(momentum=0.8))
            layer_features = max(int(layer_features * 0.5), self.min_generator_feature_size)
        model.add(Conv2D(self.img_shape[-1], kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        # model.add(Conv2D(
        #     self.train_generator.batch_size, kernel_size=3, strides=2,
        #     input_shape=self.img_shape, padding="same"
        # ))
        model.add(Conv2D(
            self.train_generator.batch_size, kernel_size=3,
            input_shape=self.img_shape, padding="same"
        ))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(256, kernel_size=3, padding="same"))
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

    def train(self, steps, sample_interval=50):
        # Adversarial ground truths
        valid = np.ones((self.train_generator.batch_size, 1))
        fake = np.zeros((self.train_generator.batch_size, 1))

        for steps in range(self.step_offset, steps + self.step_offset):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            ground_truth_images, y = self.train_generator.next()
            ground_truth_images -= 1
            if y.shape[0] != self.train_generator.batch_size:
                print(f"Got non batch size: {y.shape[0]}")
                continue

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.train_generator.batch_size, self.latent_dim))
            synthetic_images = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y, num_classes=self.num_classes + 1)
            fake_labels = to_categorical(
                np.full((self.train_generator.batch_size, 1), self.num_classes), num_classes=self.num_classes + 1
            )

            # Train the discriminator
            # x_combined = np.concatenate((ground_truth_images[:len(ground_truth_images) // 2], synthetic_images))
            # fakeness_combined = np.concatenate((valid[:len(valid) // 2], fake[:len(fake) // 2]))
            # labels_combined = np.concatenate((labels[:len(labels) // 2], fake_labels[:len(fake_labels) // 2]))
            #
            # d_loss = self.discriminator.train_on_batch(
            #     x_combined, [fakeness_combined, labels_combined], class_weight=[self.cw1, self.cw2]
            # )
            d_loss_real = self.discriminator.train_on_batch(
                ground_truth_images, [valid, labels], class_weight=[self.cw1, self.cw2]
            )
            d_loss_fake = self.discriminator.train_on_batch(
                synthetic_images, [fake, fake_labels], class_weight=[self.cw1, self.cw2]
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Clip critic weights
            for l in self.discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[self.cw1, self.cw2])

            self.write_log(['g_loss'], [g_loss], steps)
            self.write_log(['d_loss', 'ukn_loss_1', 'ukn_loss_2', 'acc', 'op_acc'], d_loss, steps)

            # Plot the progress
            print(
                f"{steps} [D loss: {d_loss[0]}, ukn_1: {d_loss[1]}, ukn_2: {d_loss[2]}, "
                f"acc: {100 * d_loss[3]}%, "
                f"op_acc: {100 * d_loss[4]}%] [G loss: {g_loss}]"
            )

            # If at save interval => save generated image samples
            if steps % sample_interval == 0:
                self.sample_images(steps)
                if steps % 5000 == 0:
                    self.save_model(steps)

    def sample_images(self, step):
        gen_imgs = self.generator.predict(self.img_save_noise)
        gen_imgs += 1
        gen_imgs *= 127.5

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
        plt.savefig(f"images/{self.run_id}/hotnot_{step}.png")
        plt.clf()
        plt.close()

    def save_model(self, step):

        def save(model, model_name):
            file_name = f"models/{self.run_id}/{model_name}.h5.tmp"
            # model.save(f"models/{self.run_id}/{model_name}.h5")
            model.save_weights(file_name)
            os.rename(file_name, file_name.replace(".tmp", ""))

        save(self.generator, f"generator/sgan_generator_{step}")
        save(self.discriminator, f"discriminator/sgan_discriminator_{step}")
        save(self.combined, f"combined/sgan_adversarial_{step}")

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
    # sgan = SGAN(_run_id="9CI8QDWY")
    sgan.train(steps=20000, sample_interval=50)
