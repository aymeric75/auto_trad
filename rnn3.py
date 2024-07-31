import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Function to load data from three files
def load_data(file_path1, file_path2, file_path3):
    data1 = np.loadtxt(file_path1)
    data2 = np.loadtxt(file_path2)
    data3 = np.loadtxt(file_path3)
    
    X1 = data1[:, :-1]
    X2 = data2[:, :-1]
    X3 = data3[:, :-1]
    y1 = data1[:, -1]
    y2 = data2[:, -1]
    y3 = data3[:, -1]
    
    # Ensure labels are consistent
    assert np.array_equal(y1, y2) and np.array_equal(y2, y3), "Labels in all files must be the same"
    
    # Concatenate features from all files
    X = np.concatenate([X1, X2, X3], axis=1)
    y = y1
    
    # Reshape for LSTM input: (num_samples, timesteps, features)
    X = X.reshape((X.shape[0], 10, 3))  # Now we have 3 features for each timestep
    
    return X, y

# Function to build the model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        return_sequences=True,
        input_shape=(10, 3)  # Updated input shape with 3 features
    ))
    model.add(layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=256, step=32)
    ))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load your data
file_path1 = 'output1.txt'
file_path2 = 'output2.txt'
file_path3 = 'output3.txt'
X, y = load_data(file_path1, file_path2, file_path3)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=1,
    directory='my_dir',
    project_name='rnn_hyperparameter_tuning'
)

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The optimal number of units in the LSTM layer is {best_hps.get('units')}")
print(f"The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}")

# Build the best model and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Evaluate the model
eval_result = model.evaluate(X_val, y_val)
print(f"Evaluation result: {eval_result}")
