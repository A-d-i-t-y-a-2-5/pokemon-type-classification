from datetime import datetime
import os
import shutil
import pandas as pd
import tensorflow as tf

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
    def __init__(self, filename: str, path: str = "./data/images"):
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.dataframe = pd.read_csv(filename, converters={"Number": str})
        self.path = path

    def create_classes(self):

        dex_primary_type = self.dataframe[["Number", "Type 1"]].values.tolist()
        for dex, p_type in dex_primary_type:
            dst_dir = f"{self.path}/{p_type.lower()}"
            os.makedirs(dst_dir, exist_ok=True)

            src_path = f"{self.path}/{dex}.png"
            dst_path = dst_dir + f"/{dex}.png"

            shutil.move(src_path, dst_path)

    def create_dataset(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            labels="inferred",
            color_mode="rgb",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            labels="inferred",
            color_mode="rgb",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

        class_names = self.train_ds.class_names
        self.n_classes = len(class_names)

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def train(self):
        self.create_dataset()
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(18, activation="softmax"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"],
        )

        # Create a TensorBoard callback
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs, histogram_freq=1, write_images=True
        )

        model.fit(
            self.train_ds,
            epochs=2,
            validation_data=self.val_ds,
            callbacks=[tboard_callback],
        )
