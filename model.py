import os
import json
import wandb
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from wandb.keras import WandbCallback
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import to_categorical

def load_data(path):
    data = np.load(path, allow_pickle=True)
    x = np.empty((len(data), 1024))
    y = np.empty((len(data), 2))

    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = to_categorical(data[i][1], num_classes=2)

    return x, y

class CoughDetect(object):
    def __init__(self, name, config, hyper=False, hyper_project="", extra=None):
        self.name = name
        if hyper:
            wandb.init(config=config, project=hyper_project)
            self.config = wandb.config
            wandb.run.save()
            try:
                os.system("mkdir sweep/"+wandb.run.name)
            except:
                pass
        else:
            self.config = config
            log_dir = "logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def transfer(self):
        # Attach custom final layers
        input_layer = layers.Input(shape=(1024,))
        x = layers.Dense(self.config["dense_1"], activation='relu')(input_layer)
        x = layers.Dropout(self.config["drop_1"])(x)
        out = layers.Dense(2, activation='softmax')(x)

        # Build model
        self.model = Model(name='VoiceMed-CoughDetect', inputs=input_layer, outputs=out)

    def train(self, x_train, y_train, validation):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["lr"],
            beta_1=self.config["beta_1"],
            beta_2=self.config["beta_2"]
        )

        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )

        self.model.summary()

        self.model.fit(
            x_train,
            y_train,
            validation_data=validation,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                WandbCallback(),
                tf.keras.callbacks.ModelCheckpoint(filepath="sweep/"+wandb.run.name+"/best.h5", monitor='val_loss', save_best_only=True)
            ]
        )

    def test(self, x_test, y_test, extra=True):
        test_err, test_acc = self.model.evaluate(x_test, y_test, verbose=0)

        if extra:
            return test_acc
        else:
            print("Accuracy on testing data: "+str(test_acc))

    def save(self):
        folder = "sweep/"+wandb.run.name+"/"
        with open(folder+"model.json", "w") as json_file:
            json_file.write(self.model.to_json())

        self.model.save_weights(folder+"model.h5")
        print("Saved model '"+self.name+"-"+wandb.run.name+"' to disk")


if __name__ == '__main__':
    config = dict(
        dense_1 = 512,
        drop_1 = 0.5,
        batch_size = 16,
        epochs = 60,

        lr = 1e-4,
        beta_1 = 0.99,
        beta_2 = 0.999,
        l2_rate = 0.001,
        alpha = 0.1
    )
    x_train, y_train = load_data("train.npy")
    x_test, y_test = load_data("test.npy")
    m = CoughDetect("TransferV0.1", config, hyper=True, hyper_project="VoiceMed")
    m.transfer()
    m.train(x_train, y_train, (x_test, y_test))
    m.save()
