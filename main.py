import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import logging
import os

#### SOURCE: https://github.com/keras-team/keras-tuner/issues/122


def list_files_in_subdirectory(directory_path):
    # Loop through each subdirectory and file in the given directory path
    data_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            data_files.append(file)
    return data_files



logging.basicConfig(filename="output.log", level=logging.INFO, format='%(asctime)s %(message)s')


for data_file in list_files_in_subdirectory("data"):




    # Set up logging to a     
    data_file_no_ext = data_file[:-4]

    print("data_file_no_ext")
    print(data_file_no_ext)
    exit()

    # Load data from the file
    data = np.loadtxt("data/"+data_file)
    X = data[:, :-1]

    n_features = X.shape[1]


    y = data[:, -1]

    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            model = keras.Sequential()
            model.add(layers.Input(shape=(n_features,)))
            model.add(layers.Dense(units=hp.Int('units1', min_value=2, max_value=9, step=1), activation=hp.Choice('activation1', values=['relu', 'sigmoid'])))
            model.add(layers.Dense(units=hp.Int('units2', min_value=3, max_value=8, step=1), activation=hp.Choice('activation2', values=['relu', 'sigmoid'])))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Float('learning_rate', min_value=1e-3, max_value=9e-1, step=0.001)),
                loss="binary_crossentropy", 
                metrics=["accuracy"],
            )

            return model

        def fit(self, hp, model, *args, **kwargs):
            return model.fit(
                *args,
                batch_size=hp.Int('batch_size', min_value=1, max_value=100, step=5), #hp.Choice("batch_size", [1, 5, 10, 15, 30]),
                **kwargs,
            )

    # RandomSearch
    tuner = kt.BayesianOptimization(
        MyHyperModel(),
        objective="val_accuracy",
        max_trials=150,
        beta=3,
        #alpha=1e-1,
        overwrite=True,
        directory=f"my_dir_{data_file_no_ext}",
        project_name="tune_hypermodel",
    )


    # tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=3)])
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info(f"FOR DATASET : {data_file_no_ext}, best config is:")
    logging.info(f"The optimal number of units in the first hidden layer is: {best_hps.get('units1')}")
    logging.info(f"The optimal number of units in the second hidden layer is: {best_hps.get('units2')}")
    logging.info(f"The optimal activation fct for the first hidden layer is: {best_hps.get('activation1')}")
    logging.info(f"The optimal activation fct for the second hidden layer is: {best_hps.get('activation2')}")
    logging.info(f"The optimal learning rate for the optimizer is: {best_hps.get('learning_rate')}")
    logging.info(f"The optimal batch size is: {best_hps.get('batch_size')}")
    

    # Train the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=100, batch_size=best_hps.get('batch_size'), validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=5)])

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    logging.info(f"Validation accuracy: {val_accuracy}")
    logging.info("")