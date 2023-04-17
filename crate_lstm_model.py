import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# STEP 1: First data visualization 

# Read in the data from the CSV file
df = pd.read_csv("block_27.csv", parse_dates=["day"], index_col=["day"])
# Select only the data for 2012-03-01 to 2014-02-27
df = df.loc[(df.index >= '2012-03-01') & (df.index <= '2014-02-27')]
# Resample the data to daily frequency and calculate the daily average
daily_energy_sum = df["energy_sum"].resample("D").mean()
# Create a new DataFrame with the daily average energy sum and date as index
df_energy_sum = pd.DataFrame({"Total Energy in a day KwH": daily_energy_sum})
# df_energy_sum.to_csv('block_27_preprocessed.csv', encoding='utf-8', index=True)
# Print the first 5 rows of the new DataFrame
'''print(df_energy_sum.head())'''
# Plot the resampled data
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(df_energy_sum.index, df_energy_sum.values)
ax.set_xlabel("Date")
ax.set_ylabel("Energy Average")
ax.set_title("Daily Energy Average from 2012-03-01 to 2014-02-27")
plt.show()

# STEP 2: Data PreProcessing 
# Get energy_sum date array
timesteps = daily_energy_sum.index.to_numpy()
energy_kwh = daily_energy_sum.values.flatten()

# Create train and test splits the right way for time series data
split_size = int(0.8 * len(energy_kwh)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], energy_kwh[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], energy_kwh[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)

# Plot correctly made splits
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("Daily energy kwh")
plt.legend(fontsize=14)
plt.show();


# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("energy kwh")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)

def evaluate_preds(y_true, y_pred):
  # Make sure float32 (for metric calculations)
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)


  # Account for different sized metrics (for longer horizons, reduce to single number)
  if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)


  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy()}

# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
  return x[:, :-horizon], x[:, -horizon:]


# Create function to view NumPy arrays as windows 
def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels

full_windows, full_labels = make_windows(energy_kwh, window_size=WINDOW_SIZE, horizon=HORIZON)

# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

import os

# Create a function to implement a ModelCheckpoint callback with a specific filename 
def create_model_checkpoint(model_name, save_path="models"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), # create filepath to save model
                                            verbose=0, # only output a limited amount of text
                                            save_best_only=True) # save only the best model to file

def make_preds(model, input_data):

  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions

HORIZON = 7
WINDOW_SIZE = 30

full_windows, full_labels = make_windows(energy_kwh, window_size=WINDOW_SIZE, horizon=HORIZON)

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows=full_windows, labels=full_labels, test_split=0.2)




# Building Model
# Set random seed for reproducibility
tf.random.set_seed(42)


# Define the learning rate and early stopping patience
LR = 0.001
PATIENCE = 5

# Define the input layer with the shape of the window size
inputs = layers.Input(shape=(WINDOW_SIZE))

# Add a Lambda layer to expand the dimension of the input tensor to make it compatible with the LSTM layer
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

# Add the first LSTM layer with 128 units and ReLU activation function
x = layers.LSTM(128, activation="relu")(x)

# Add three additional Dense layers with 64, 32, and 16 units respectively, with tanh, sigmoid, and ReLU activation functions
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="sigmoid")(x)

# Define the output layer with the same number of units as the horizon
output = layers.Dense(HORIZON)(x)

# Define the model with the input and output layers
model_13 = tf.keras.Model(inputs=inputs, outputs=output, name="model_13_lstm")

# Define the optimizer with the specified learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# Compile the model with MAE loss and the optimizer
model_13.compile(loss="mae", optimizer=optimizer)

# Define the early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')

# Fit the model with early stopping callback
history = model_13.fit(train_windows,
                    train_labels,
                    epochs=100,
                    verbose=1,
                    batch_size=32,
                    validation_data=(test_windows, test_labels),
                    callbacks=[early_stopping, create_model_checkpoint(model_name=model_13.name)])

# Model Evaluation 
model_13 = tf.keras.models.load_model("model_experiments/model_13_lstm/")
model_13.evaluate(test_windows, test_labels)

# Make predictions with our LSTM model
model_13_preds = make_preds(model_13, test_windows)
model_13_preds[:1]

# Evaluate model preds
model_12_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_13_preds)
model_12_results

plt.figure(figsize=(10, 7))
# Plot model_3_preds by aggregating them (note: this condenses information so the preds will look fruther ahead than the test data)
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=test_labels[:, 0], 
                 label="Test_data")
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=tf.reduce_mean(model_13_preds, axis=1), 
                 format="-",
                 label="model_13_preds")
