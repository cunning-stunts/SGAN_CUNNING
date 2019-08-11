import os

from tensorflow.python.data.experimental import AUTOTUNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)
import numpy as np
from tensorflow.python.ops.image_ops_impl import convert_image_dtype, ResizeMethod

from consts import INPUT_IMG_SHAPE, OUTPUT_IMG_SHAPE, BATCH_SIZE, CROP_SIZE, CROP
from rxrx1_df import get_dataframe
from utils import get_number_of_target_classes


def load_img(feature, label):
    final_tensor = None
    channels = range(1, 7)
    for channel in channels:
        img_path_title = f'img_loc_{channel}'
        path = feature[img_path_title]
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=INPUT_IMG_SHAPE[-1])
        image.set_shape(INPUT_IMG_SHAPE)
        if final_tensor is None:
            final_tensor = image
        else:
            final_tensor = tf.concat([final_tensor, image], axis=2)

    for i in channels:
        del feature[f"img_loc_{i}"]
    feature["img"] = final_tensor
    return feature, label


def add_gausian_noise(x_new, std_dev):
    dtype = x_new.dtype
    flt_image = convert_image_dtype(x_new, tf.dtypes.float32)
    flt_image += tf.random_normal(shape=tf.shape(flt_image), mean=0, stddev=std_dev, dtype=tf.dtypes.float32)
    return tf.image.convert_image_dtype(flt_image, dtype, saturate=True)


def img_augmentation(x_dict, label):
    x = x_dict["img"]
    x_new = tf.image.random_brightness(x, 0.05)
    x_new = tf.image.random_contrast(x_new, 0.8, 1.2)
    x_new = tf.image.random_flip_left_right(x_new)
    x_new = tf.image.random_flip_up_down(x_new)
    # x_new = tf.image.random_hue(x_new, 0.06) # requires colour
    # x_new = tf.image.random_saturation(x_new, 0.1, 1.9) # requires colour
    x_new = add_gausian_noise(x_new, std_dev=0.05)
    x_dict["img"] = x_new

    return x_dict, label


def normalise_image(x_dict, label):
    image = x_dict["img"]
    image = tf.image.per_image_standardization(image)
    if not CROP:
        image = tf.image.resize_images(
            image, [OUTPUT_IMG_SHAPE[0], OUTPUT_IMG_SHAPE[1]], method=ResizeMethod.AREA
        )
    x_dict["img"] = image
    return x_dict, label


def crop_image(x_dict, label):
    x_dict["img"] = tf.image.random_crop(
        x_dict["img"],
        size=CROP_SIZE
    )
    return x_dict, label


def get_ds(
        df, number_of_target_classes, training=False,
        shuffle_buffer_size=10_000,
        shuffle=None, normalise=True,
        perform_img_augmentation=None
):
    if shuffle is None:
        shuffle = True if training else False
    if perform_img_augmentation is None:
        perform_img_augmentation = True if training else False
    one_hot = tf.one_hot(df.pop("sirna"), number_of_target_classes)

    ds = tf.data.Dataset.from_tensor_slices((dict(df), one_hot))
    ds = ds.map(
        map_func=load_img,
        num_parallel_calls=AUTOTUNE
    )
    # ds = ds.apply(tf.data.experimental.ignore_errors())
    if CROP:
        ds = ds.map(
            map_func=crop_image,
            num_parallel_calls=AUTOTUNE
        )
    if perform_img_augmentation:
        ds = ds.map(
            map_func=img_augmentation,
            num_parallel_calls=AUTOTUNE
        )
    if shuffle:
        print(f"Filling shuffle buffer {shuffle_buffer_size}, this may take some time...")
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if normalise:
        ds = ds.map(
            map_func=normalise_image,
            num_parallel_calls=AUTOTUNE
        )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds.repeat()


def show_ds(ds):
    iter = ds.make_one_shot_iterator()
    x, y = iter.get_next()
    with tf.Session() as sess:
        while True:
            try:
                x1, y1 = sess.run([x, y])
                imgs = x1["img"]
                for img, y2 in zip(imgs, y1):  # batch size
                    #
                    # TODO:
                    # BEAR IN MIND
                    # THE ACTUAL CLASS ID IS 1..N
                    # BUT ARGMAX WILL RETURN 0..N-1
                    # TODO:
                    #
                    argmax = np.argmax(y2)
                    print(f"y2: {argmax}")
                    cv2.imshow('image', img)
                    cv2.waitKey(0)
            except Exception as e:
                print(e)
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _df = get_dataframe("D:\\rxrx1")
    number_of_classes = get_number_of_target_classes(_df)
    _ds = get_ds(
        _df, number_of_target_classes=number_of_classes, normalise=False
    )
    show_ds(_ds)
