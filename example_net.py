import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import time

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard

from consts import BATCH_SIZE, EPOCHS, EMBEDDING_DIMS, HASH_BUCKET_SIZE, HIDDEN_UNITS, SHUFFLE_BUFFER_SIZE, \
    TENSORBOARD_UPDATE_FREQUENCY, OUTPUT_IMG_SHAPE
from rxrx1_df import get_dataframe
from rxrx1_ds import get_ds
from utils import get_random_string, get_number_of_target_classes


def wide_and_deep_classifier(
        inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units, number_of_target_classes, img_input
):
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
    for layerno, numnodes in enumerate(dnn_hidden_units):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno + 1))(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)

    # img_net = tf.keras.applications.mobilenet_v2.MobileNetV2(
    #     input_shape=OUTPUT_IMG_SHAPE,
    #     alpha=1.0,
    #     include_top=False,
    #     weights=None,
    #     input_tensor=img_input,
    #     pooling="max",
    #     classes=number_of_target_classes,
    # )
    # both = tf.keras.layers.concatenate([deep, wide, img_net.output], name='both')

    both = tf.keras.layers.concatenate([deep, wide], name='both')
    output = tf.keras.layers.Dense(number_of_target_classes, activation='softmax', name='pred')(both)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_features(ds):
    real = {name: tf.feature_column.numeric_column(name)
            for name, dtype in ds.output_types[0].items()
            if name != "img" and dtype.name in ["int64"]}
    sparse = {name: tf.feature_column.categorical_column_with_hash_bucket(name, hash_bucket_size=HASH_BUCKET_SIZE)
              for name, dtype in ds.output_types[0].items()
              if dtype.name in ["string"]}
    inputs = {colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
              for colname in real.keys()}
    inputs.update({colname: tf.keras.layers.Input(name=colname, shape=(), dtype='string')
                   for colname in sparse.keys()})
    # embed all the sparse columns
    embed = {'embed_{}'.format(colname): tf.feature_column.embedding_column(col, EMBEDDING_DIMS)
             for colname, col in sparse.items()}
    real.update(embed)
    # one-hot encode the sparse columns
    sparse = {colname: tf.feature_column.indicator_column(col)
              for colname, col in sparse.items()}

    img_input = tf.keras.layers.Input(name="img", shape=OUTPUT_IMG_SHAPE, dtype='float32')
    return inputs, sparse, real, img_input


def train_model(model, train_ds, test_ds, run_id, steps_per_epoch, validation_steps_per_epoch):
    model_path = os.path.join("models", run_id)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=os.path.join(model_path, 'model.cpt'),
        save_weights_only=False,
        save_best_only=True,
        verbose=1,
        load_weights_on_restart=True
    )
    callback = TensorBoard(model_path, update_freq=TENSORBOARD_UPDATE_FREQUENCY)
    callback.set_model(model)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps_per_epoch,
        callbacks=[cp_callback, callback]
    )
    return history


def export_saved_model(run_id, model):
    export_dir = os.path.join('models', run_id, f'model_{time.strftime("%Y%m%d-%H%M%S")}')
    print('Exporting to {}'.format(export_dir))
    tf.keras.experimental.export_saved_model(model, export_dir)


def main(_run_id=None):
    df = get_dataframe("D:\\rxrx1")
    number_of_target_classes = get_number_of_target_classes(df)

    if _run_id is None:
        run_id = get_random_string(8)
    else:
        run_id = _run_id

    # no need for ID
    df.pop("id_code")

    # these aren't in test dataset :(
    #   we could use them, then when we are running inference on test data:
    #       use random values
    #       use average values
    df.pop("site_num")
    df.pop("microscope_channel")
    df.pop("well_type")

    train_df, test_df = train_test_split(df)
    training_samples = len(train_df.index)
    training_steps_per_epoch = training_samples // BATCH_SIZE
    validation_samples = len(test_df.index)
    validation_steps_per_epoch = validation_samples // BATCH_SIZE

    print(f"""
    number_of_target_classes: {number_of_target_classes}
    total_samples: {training_samples + validation_samples}
    training_samples: {training_samples}
    training_steps_per_epoch: {training_steps_per_epoch}
    validation_samples: {validation_samples}
    validation_steps_per_epoch: {validation_steps_per_epoch}
    run_id: {run_id}
    gpu: {tf.test.is_gpu_available()}
    cuda: {tf.test.is_built_with_cuda()}
    """)

    train_ds = get_ds(
        train_df, number_of_target_classes=number_of_target_classes,
        training=True, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        normalise=False, perform_img_augmentation=False
    )
    test_ds = get_ds(
        test_df, number_of_target_classes=number_of_target_classes,
        normalise=False
    )

    inputs, sparse, real, img_input = get_features(train_ds)
    model = wide_and_deep_classifier(
        inputs,
        linear_feature_columns=sparse.values(),
        dnn_feature_columns=real.values(),
        dnn_hidden_units=HIDDEN_UNITS,
        number_of_target_classes=number_of_target_classes,
        img_input=img_input
    )

    # tf.keras.utils.plot_model(model, f'models/{run_id}/model.png', show_shapes=True, rankdir='LR')
    train_model(model, train_ds, test_ds, run_id, training_steps_per_epoch, validation_steps_per_epoch)
    export_saved_model(run_id, model)
    print("")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _run_id = sys.argv[1]
    else:
        _run_id = None
    main(_run_id=_run_id)
