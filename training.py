from collections import Counter
from datetime import datetime
from glob import glob
import io
import itertools
import os
from pprint import pprint
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow import keras
from tensorflow.keras import layers


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Trainer:
    def __init__(self, filename: str, path: str = "./data/images", mode: str = "train"):
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.dataframe = pd.read_csv(filename, converters={"Number": str})
        self.mode = mode
        self.path = path + "/" + self.mode
        self.test_path = path + "/" + "test"

    def class_indices(self):
        type_training_paths = glob(f"{self.path}/*")
        type_classes = sorted([path.split("/")[-1] for path in type_training_paths])
        self.type_lookup = layers.StringLookup(vocabulary=type_classes)
        # self.class_indices_dict = dict(zip(type_classes, range(len(type_classes))))

    def stratified_split(self):
        X = glob(f"{self.path}/*/*.png")
        y = list(map(lambda x: x.split("/")[-2], X))
        # skf = StratifiedKFold(n_splits=3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=0.2
        )
        # print(pd.DataFrame(y_train).value_counts(normalize=True))
        # print(pd.DataFrame(y_test).value_counts(normalize=True))
        # pprint(Counter(y_train))
        # pprint(Counter(y_test))

    def make_ds(self, path):
        BUFFER_SIZE = 256
        # print(path)
        # ds = tf.data.Dataset.list_files(path + "/*.png", shuffle=False)
        ds = tf.data.Dataset.from_tensor_slices(path)
        # print(ds)
        ds = ds.shuffle(BUFFER_SIZE)
        return ds

    def process_path(self, file_path):
        # print(file_path, os.path.sep)
        parts = tf.strings.split(file_path, os.path.sep)
        # parts = file_path.split(os.path.sep)
        # print(parts)

        # type_val = tf.keras.backend.get_value(parts[-2])
        label = self.type_lookup(parts[-2]) - 1

        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_height, self.img_width])

        return img, label

    def oversample(self, X, y):
        """
        Returns a list of Dataset objects for each class
        """
        # type_training_paths = glob(f"{self.path}/*")
        # type_ds_list = [self.make_ds(path) for path in type_training_paths]
        # return type_ds_list
        X = np.array(X)
        y = np.array(y)
        unique_classes = np.unique(y)
        type_ds_list = []
        for type_cls in unique_classes:
            # Get indices of the current class
            indices = np.where(y == type_cls)[0]
            X_class = X[indices]
            # y_class = y[indices]
            dataset = tf.data.Dataset.from_tensor_slices(X_class)
            # print(len(X_class))
            dataset = dataset.shuffle(buffer_size=len(X_class)).repeat()
            type_ds_list.append(dataset)
        return type_ds_list
        # pprint(type_training_paths)

    def create_classes(self):

        dex_primary_type = self.dataframe[["Number", "Type 1"]].values.tolist()
        for dex, p_type in dex_primary_type:
            dst_dir = f"{self.path}/{p_type.lower()}"
            os.makedirs(dst_dir, exist_ok=True)

            src_path = f"{self.path}/{dex}.png"
            dst_path = dst_dir + f"/{dex}.png"

            shutil.move(src_path, dst_path)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def create_dataset(self):
        # self.train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        #     self.path,
        #     labels="inferred",
        #     color_mode="rgb",
        #     validation_split=0.2,
        #     subset="training",
        #     seed=123,
        #     image_size=(self.img_height, self.img_width),
        #     batch_size=self.batch_size,
        # )

        # self.val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        #     self.path,
        #     labels="inferred",
        #     color_mode="rgb",
        #     validation_split=0.2,
        #     subset="validation",
        #     seed=123,
        #     image_size=(self.img_height, self.img_width),
        #     batch_size=self.batch_size,
        # )

        self.test_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_path,
            labels="inferred",
            color_mode="rgb",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        # self.class_names = self.train_ds.class_names
        # self.n_classes = len(class_names)

        # count = np.zeros(self.n_classes, dtype=np.int32)
        # for _, labels in self.train_ds:
        #     y, _, c = tf.unique_with_counts(labels)
        #     count[y.numpy()] += c.numpy()

        # self.class_weights = {
        #     i: (1 / (count.sum() - count[i])) * (count.sum() / 2.0)
        #     for i in range(self.n_classes)
        # }

        # AUTOTUNE = tf.data.AUTOTUNE
        # self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def train_test_model(self, run_dir):
        # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=run_dir, histogram_freq=1, write_images=True
        )
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        cm_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=self.log_confusion_matrix
        )

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip(
                    "horizontal", input_shape=(self.img_height, self.img_width, 3)
                ),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )
        self.model = tf.keras.Sequential(
            [
                data_augmentation,
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 5, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 5, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 5, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(18, activation="softmax"),
            ]
        )
        self.model.summary()

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(1e-5),
            metrics=["accuracy"],
        )
        self.model.fit(
            self.train_ds,
            epochs=100,
            validation_data=self.val_ds,
            callbacks=[tboard_callback, callback, cm_callback],
            # class_weight=self.class_weights
        )
        _, accuracy = self.model.evaluate(self.test_ds)

        return accuracy

    def run(self, run_dir):
        accuracy = self.train_test_model(run_dir)
        print(accuracy)

    def train(self):
        self.stratified_split()
        self.class_indices()
        self.train_ds = (
            self.resample_dataset(self.X_train, self.y_train)
            .batch(self.batch_size)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        self.val_ds = (
            self.resample_dataset(self.X_test, self.y_test)
            .batch(self.batch_size)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        # count = 0
        # for i,j in self.train_ds:
        #     count += j.shape[0]
        #     # break
        # print(count)
        # return
        # for i in type_dataset_list[0]:
        #     print(i.numpy().decode("utf-8"))
        #     # print(type(i))
        #     break

        # type_ds = [ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE) for ds in type_dataset_list]
        self.create_dataset()
        self.run("logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    def resample_dataset(self, X, y):
        type_dataset_list = self.oversample(X, y)
        # print(len(type_dataset_list))
        type_ds = [
            td.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE).take(256)
            for td in type_dataset_list
        ]
        resampled_ds = tf.data.Dataset.sample_from_datasets(
            type_ds, weights=[1 / (len(type_ds))] * (len(type_ds))
        )
        return resampled_ds

        # Create a TensorBoard callback
        # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1)
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2
        )

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        return figure

    def log_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.test_ds)
        test_pred = np.argmax(test_pred_raw, axis=1)

        logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        test_labels = np.concatenate([y for x, y in self.test_ds], axis=0)
        file_writer_cm = tf.summary.create_file_writer(logdir + "/cm")

        # Calculate the confusion matrix.
        cm = metrics.confusion_matrix(test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.type_lookup.get_vocabulary(False))
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)
