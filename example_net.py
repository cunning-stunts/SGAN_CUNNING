import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Flatten, Dense

from rxrx1_ds import get_dataframe


def main():
    df = get_dataframe("D:\\rxrx1")
    targets = df["sirna"].tolist()
    unique_classes = set(targets)

    df.pop("id_code")
    train, test = train_test_split(df)

    train_ds = get_ds(train)
    test_ds = get_ds(test, shuffle=False)

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
    image = tf.image.decode_png(image, channels=1)
    feature["img_location"] = image
    return feature, label


def get_ds(df, shuffle=True, batch_size=32):
    target = df.pop("sirna")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    ds = ds.map(load_img)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).repeat()
    return ds


if __name__ == '__main__':
    main()
