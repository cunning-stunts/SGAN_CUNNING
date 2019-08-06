import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.ops.image_ops_impl import ResizeMethod, convert_image_dtype

from rxrx1_ds import get_dataframe

INPUT_IMG_SHAPE = (512, 512, 1)
OUTPUT_IMG_SHAPE = (512, 512, 1)


def main():
    df = get_dataframe("D:\\rxrx1")
    targets = df["sirna"].tolist()
    unique_classes = set(targets)

    df.pop("id_code")
    train, test = train_test_split(df)

    train_ds = get_ds(train, training=True)
    test_ds = get_ds(test)

    iter = train_ds.make_one_shot_iterator()
    x, y = iter.get_next()
    with tf.Session() as sess:
        x1, y1 = sess.run([x, y])
        print(x1, y1)

    input_shape = 123
    num_classes = len(unique_classes)

    model = get_compiled_model(num_classes)

    model.fit(
        train_ds,
        epochs=10,
        steps_per_epoch=30,
        validation_data=test_ds,
        validation_steps=3
    )

    print("")


def get_compiled_model(num_classes):
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(512, 512, 3),
        alpha=1.0,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None
    )
    x = model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def load_img(feature, label):
    path = feature["img_location"]
    image = tf.read_file(path)
    image = tf.io.decode_image(image, channels=INPUT_IMG_SHAPE[-1])
    image.set_shape(INPUT_IMG_SHAPE)
    del feature["img_location"]
    feature["img"] = image
    return feature, label


def add_gausian_noise(x_new):
    dtype = x_new.dtype
    flt_image = convert_image_dtype(x_new, tf.dtypes.float32)
    flt_image += tf.random_normal(shape=tf.shape(flt_image), mean=0, stddev=.1, dtype=tf.dtypes.float32)
    return tf.image.convert_image_dtype(flt_image, dtype, saturate=True)


def img_augmentation(x_dict, label):
    x = x_dict["img"]
    x_new = tf.image.random_brightness(x, 0.5)
    x_new = tf.image.random_contrast(x_new, 0.4, 1.4)
    # x_new = tf.image.random_hue(x_new, 0.06) # requires colour
    # x_new = tf.image.random_saturation(x_new, 0.1, 1.9) # requires colour
    x_new = add_gausian_noise(x_new)
    x_dict["img"] = x_new

    return x_dict, label


def normalise_image(x_dict, label):
    image = x_dict["img"]
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(
        image, (OUTPUT_IMG_SHAPE[0], OUTPUT_IMG_SHAPE[1]), method=ResizeMethod.AREA
    )
    x_dict["img"] = image
    return x_dict, label


def get_ds(
        df, training=False, batch_size=32,
        shuffle_buffer_size=10_000,
        shuffle=None
):
    if shuffle is None:
        shuffle = True if training else False
    target = df.pop("sirna")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    ds = ds.map(
        load_img,
        num_parallel_calls=os.cpu_count()
    )
    if training:
        ds = ds.map(
            map_func=img_augmentation,
            num_parallel_calls=os.cpu_count()
        )
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.apply(tf.data.experimental.map_and_batch(
        map_func=normalise_image,
        batch_size=batch_size,
        num_parallel_batches=os.cpu_count(),
    ))
    return ds


if __name__ == '__main__':
    main()
