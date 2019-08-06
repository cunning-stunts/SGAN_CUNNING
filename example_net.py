import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
import time

import matplotlib.pyplot as plt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard

from consts import BATCH_SIZE, EPOCHS, EMBEDDING_DIMS, HASH_BUCKET_SIZE, HIDDEN_UNITS, SHUFFLE_BUFFER_SIZE
from rxrx1_df import get_dataframe
from rxrx1_ds import get_ds
from utils import get_random_string, get_number_of_target_classes


def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
    for layerno, numnodes in enumerate(dnn_hidden_units):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno + 1))(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)
    both = tf.keras.layers.concatenate([deep, wide], name='both')
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
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
    return inputs, sparse, real


def train_model(model, train_ds, test_ds, run_id, steps_per_epoch):
    model_path = os.path.join("models", run_id)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_path, 'model.cpt'),
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )
    callback = TensorBoard(model_path, update_freq='batch')
    callback.set_model(model)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[cp_callback, callback]
    )
    return history


def plot_history(history):
    print(history.history.keys())
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(10, 5))

    for idx, key in enumerate(['loss', 'accuracy']):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')


def export_saved_model(run_id, model):
    export_dir = f'models/{run_id}/export/model_{time.strftime("%Y%m%d-%H%M%S")}'
    print('Exporting to {}'.format(export_dir))
    tf.keras.experimental.export_saved_model(model, export_dir)


def main(_run_id=None):
    df = get_dataframe("D:\\rxrx1")
    number_of_target_classes = get_number_of_target_classes(df)
    total_samples = len(df.index)
    steps_per_epoch = total_samples // BATCH_SIZE

    if _run_id is None:
        run_id = get_random_string(8)
    else:
        run_id = _run_id

    print(f"""
    number_of_target_classes: {number_of_target_classes}
    total_samples: {total_samples}
    steps_per_epoch: {steps_per_epoch}
    run_id: {run_id}
    gpu: {tf.test.is_gpu_available()}
    cuda: {tf.test.is_built_with_cuda()}
    """)

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
    train_ds = get_ds(
        train_df, number_of_target_classes=number_of_target_classes,
        training=True, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE
    )
    test_ds = get_ds(test_df, number_of_target_classes=number_of_target_classes)

    inputs, sparse, real = get_features(train_ds)
    model = wide_and_deep_classifier(
        inputs,
        linear_feature_columns=sparse.values(),
        dnn_feature_columns=real.values(),
        dnn_hidden_units=HIDDEN_UNITS
    )

    # tf.keras.utils.plot_model(model, f'models/{run_id}/model.png', show_shapes=True, rankdir='LR')
    history = train_model(model, train_ds, test_ds, run_id, steps_per_epoch)
    plot_history(history)
    export_saved_model(run_id, model)
    print("")


if __name__ == '__main__':
    main()
