# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:13:14 2024

@author: EdisonLee
"""

# path_prediction.py

import os
import pandas as pd
import numpy as np
import warnings
import re
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy.special import softmax

def load_training_data(data_folder):
    """
    Load and preprocess training data from the specified folder.
    """
    all_files = []
    for folder in ['train', 'val', 'test']:
        folder_path = os.path.join(data_folder, folder)
        all_files.extend(glob.glob(os.path.join(folder_path, "*.txt")))

    data_list = []

    for filename in all_files:
        df = pd.read_csv(filename, sep='\s+', header=None)
        df.columns = ['Timestamps', 'subclass', 'x_center', 'y_center',
                      'z_center', 'x_length', 'y_length', 'z_length']
        data_list.append(df)

    train_data = pd.concat(data_list, ignore_index=True)
    return train_data

def preprocess_data(df):
    """
    Preprocess the data for training the LSTM model.
    """
    # Sort by subclass and Timestamps
    df = df.sort_values(by=['subclass', 'Timestamps'])

    # Remove subclasses with zero initial lengths to prevent division by zero
    zero_length_subclasses = df.groupby('subclass').filter(
        lambda x: (x['x_length'].iloc[0] == 0) or
                  (x['y_length'].iloc[0] == 0) or
                  (x['z_length'].iloc[0] == 0)
    )['subclass'].unique()

    if len(zero_length_subclasses) > 0:
        print(f"Removing subclasses with zero initial lengths: {zero_length_subclasses}")
        df = df[~df['subclass'].isin(zero_length_subclasses)]

    # Add a small epsilon to prevent division by zero
    epsilon = 1e-8

    # Normalize positions and sizes relative to the first position and size of each subclass
    df['x_center_norm'] = df['x_center'] - df.groupby('subclass')['x_center'].transform('first')
    df['y_center_norm'] = df['y_center'] - df.groupby('subclass')['y_center'].transform('first')
    df['z_center_norm'] = df['z_center'] - df.groupby('subclass')['z_center'].transform('first')

    df['x_length_norm'] = df['x_length'] / (df.groupby('subclass')['x_length'].transform('first') + epsilon)
    df['y_length_norm'] = df['y_length'] / (df.groupby('subclass')['y_length'].transform('first') + epsilon)
    df['z_length_norm'] = df['z_length'] / (df.groupby('subclass')['z_length'].transform('first') + epsilon)

    # Verify data for NaN or Inf values
    if df[['x_center_norm', 'y_center_norm', 'z_center_norm',
           'x_length_norm', 'y_length_norm', 'z_length_norm']].isnull().any().any():
        print("Found NaN values in data after normalization.")
        df = df.dropna(subset=['x_center_norm', 'y_center_norm', 'z_center_norm',
                               'x_length_norm', 'y_length_norm', 'z_length_norm'])

    if np.isinf(df[['x_center_norm', 'y_center_norm', 'z_center_norm',
                    'x_length_norm', 'y_length_norm', 'z_length_norm']].values).any():
        print("Found Inf values in data after normalization.")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['x_center_norm', 'y_center_norm', 'z_center_norm',
                               'x_length_norm', 'y_length_norm', 'z_length_norm'])

    # Prepare sequences
    subclass_groups = df.groupby('subclass')
    sequences = []

    for _, group in subclass_groups:
        features = group[['x_center_norm', 'y_center_norm', 'z_center_norm',
                          'x_length_norm', 'y_length_norm', 'z_length_norm']].values
        if len(features) >= 20:  # Minimum sequence length
            for i in range(len(features) - 19):
                seq_input = features[i:i+8]
                seq_output = features[i+8:i+20]
                sequences.append((seq_input, seq_output))

    return sequences

def build_model():
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(8, 6), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(6, activation='linear'))  # Predicting next time step (6 features)

    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(sequences, model_path):
    """
    Train the LSTM model on the provided sequences and save it.
    """
    X_train = np.array([seq[0] for seq in sequences])  # Shape: (num_samples, 8, 6)
    y_train = np.array([seq[1][0] for seq in sequences])  # Predicting the next time step

    model = build_model()

    # Define a callback to stop training if loss is NaN
    class TerminateOnNaN(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs.get('loss') is not None:
                if np.isnan(logs.get('loss')):
                    print("NaN loss detected. Stopping training.")
                    self.model.stop_training = True

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    terminate_on_nan = TerminateOnNaN()

    model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[early_stopping, terminate_on_nan])

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to '{model_path}'")

    return model

def predict_trajectories(model, agent_data, prediction_timestamps):
    """
    Predict future trajectories for a single agent using the trained LSTM model.
    Generate 3 possible paths and assign realistic confidence scores.
    """
    # Preprocess the agent data
    agent_data = agent_data.sort_values(by='Timestamps')
    
    # Normalize positions and sizes relative to the first position
    x0 = agent_data['x_center'].iloc[0]
    y0 = agent_data['y_center'].iloc[0]
    z0 = agent_data['z_center'].iloc[0]
    x_length0 = agent_data['x_length'].iloc[0] + 1e-8  # Add epsilon to prevent division by zero
    y_length0 = agent_data['y_length'].iloc[0] + 1e-8
    z_length0 = agent_data['z_length'].iloc[0] + 1e-8

    agent_data['x_center_norm'] = agent_data['x_center'] - x0
    agent_data['y_center_norm'] = agent_data['y_center'] - y0
    agent_data['z_center_norm'] = agent_data['z_center'] - z0
    agent_data['x_length_norm'] = agent_data['x_length'] / x_length0
    agent_data['y_length_norm'] = agent_data['y_length'] / y_length0
    agent_data['z_length_norm'] = agent_data['z_length'] / z_length0

    features = agent_data[['x_center_norm', 'y_center_norm', 'z_center_norm',
                           'x_length_norm', 'y_length_norm', 'z_length_norm']].values

    if len(features) < 8:
        return None  # Not enough data to make a prediction

    input_seq = features[-8:]  # Use the last 8 steps as input

    # Generate 3 possible paths
    num_samples = 3
    paths = []
    log_likelihoods = []

    for _ in range(num_samples):
        # Initialize the sequence with the input sequence
        pred_seq = input_seq.copy()
        predictions = []
        total_log_prob = 0

        for t in range(len(prediction_timestamps)):
            # Prepare input for the model: last 8 steps
            pred_input = pred_seq[-8:].reshape(1, 8, 6)
            pred = model(pred_input, training=True).numpy().reshape(6)

            # Add noise to prediction to simulate uncertainty
            pred_noise = np.random.normal(0, 0.01, size=6)
            pred += pred_noise

            # Append prediction to sequence
            pred_seq = np.vstack([pred_seq, pred])

            # Store prediction
            predictions.append(pred)

            # For log-likelihood calculation, compare pred to model's output mean
            pred_true = pred_input[0, -1]
            mse = np.mean((pred - pred_true) ** 2)
            total_log_prob -= mse  # Subtract MSE to simulate log-likelihood

        predictions = np.array(predictions)  # Shape: (num_timesteps, 6)
        paths.append(predictions)
        log_likelihoods.append(total_log_prob)

    paths = np.array(paths)  # Shape: (num_samples, num_timesteps, 6)

    # Denormalize positions and sizes
    paths[:, :, 0] += x0  # x_center
    paths[:, :, 1] += y0  # y_center
    paths[:, :, 2] += z0  # z_center
    paths[:, :, 3] *= x_length0  # x_length
    paths[:, :, 4] *= y_length0  # y_length
    paths[:, :, 5] *= z_length0  # z_length

    # Apply softmax to negative log-likelihoods to obtain normalized confidence scores
    confidence_scores = softmax(log_likelihoods)

    # **Sort paths based on confidence scores in descending order**
    sorted_indices = np.argsort(-confidence_scores)  # Indices of paths sorted by confidence (highest first)
    paths = paths[sorted_indices]
    confidence_scores = confidence_scores[sorted_indices]

    # Assign the prediction timestamps
    num_timestamps = len(prediction_timestamps)
    path_predictions = []

    for idx, i in enumerate(range(num_samples)):
        pred_features = paths[i]
        df_pred = pd.DataFrame(pred_features, columns=['x_center', 'y_center', 'z_center',
                                                       'x_length', 'y_length', 'z_length'])

        # Ensure the number of predictions matches the number of timestamps
        if len(df_pred) > num_timestamps:
            df_pred = df_pred.iloc[:num_timestamps]
        elif len(df_pred) < num_timestamps:
            # Pad the predictions
            last_row = df_pred.iloc[-1]
            padding = pd.DataFrame([last_row] * (num_timestamps - len(df_pred)), columns=df_pred.columns)
            df_pred = pd.concat([df_pred, padding], ignore_index=True)

        # Assign the prediction timestamps
        df_pred['Timestamps'] = prediction_timestamps

        # Assign path_ID based on sorted order
        df_pred['path_ID'] = idx + 1  # Paths are numbered from 1 to 3 based on confidence
        df_pred['tracking_ID'] = agent_data['tracking_id'].iloc[0] if 'tracking_id' in agent_data.columns else 0
        df_pred['subclass'] = agent_data['subclass'].iloc[0]
        df_pred['confidence_score'] = confidence_scores[i]

        # Rearrange columns
        cols = ['Timestamps', 'tracking_ID', 'path_ID', 'subclass',
                'x_center', 'y_center', 'z_center',
                'x_length', 'y_length', 'z_length', 'confidence_score']
        df_pred = df_pred[cols]

        path_predictions.append(df_pred)

    return path_predictions


def main():
    # Set the environment variable to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Suppress warnings (optional)
    warnings.filterwarnings('ignore')

    # Paths to the datasets
    data_folder = r'E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\sgan-master\sgan-master\datasets\ISC'
    train_data_folder = data_folder  # Contains train, val, test folders

    # Path to the data files for prediction
    prediction_data_folder = r'E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\tracking_v3_merged_withID\tracking'

    # Path to the prediction timestamps
    timestamps_folder = r'E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\prediction_timestamps'

    # Output folder for the predicted trajectories
    output_folder = r'E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\predicted_trajectories_v3'
    os.makedirs(output_folder, exist_ok=True)

    # Path to save/load the model
    model_path = r'E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\saved_model\trajectory_model.h5'

    # Check if the model already exists
    if os.path.exists(model_path):
        print("Loading the saved model...")
        model = load_model(model_path, compile=False)
        # Recompile the model to ensure it's ready for prediction
        model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='mse')
        print("Model loaded successfully.")
    else:
        # Load and preprocess training data
        print("Loading training data...")
        train_data = load_training_data(train_data_folder)
        sequences = preprocess_data(train_data)
        print(f"Number of training sequences: {len(sequences)}")

        # Train and save the model
        print("Training the model...")
        model = train_model(sequences, model_path)
        print("Model training completed.")

    # Process each file in the prediction data folder
    files_processed = 0
    for filename in os.listdir(prediction_data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(prediction_data_folder, filename)
            print(f'Processing file: {filename}')

            # Extract run number without padding
            match = re.search(r'Run_(\d+)', filename)
            if match:
                run_number = match.group(1)
            else:
                run_number = str(files_processed + 1)  # Fallback if run number not found

            # Read the prediction timestamps
            timestamps_file = os.path.join(
                timestamps_folder,
                f'Run_{run_number}',
                f'Run_{run_number}_Prediction_Timstamps.csv'
            )

            if not os.path.exists(timestamps_file):
                print(f"Prediction timestamps file not found for Run_{run_number}. Skipping.")
                continue

            prediction_timestamps = pd.read_csv(timestamps_file, header=None)[0].values

            # Read the data file
            df = pd.read_csv(file_path)

            # Collect all predictions for this file
            file_predictions = []

            # Get unique tracking IDs and subclasses
            if 'tracking_id' in df.columns:
                agents = df[['tracking_id', 'subclass']].drop_duplicates()
            else:
                df['tracking_id'] = 0  # Assign a default tracking ID if not present
                agents = df[['tracking_id', 'subclass']].drop_duplicates()

            # Process each agent
            for _, agent in agents.iterrows():
                tracking_id = agent['tracking_id']
                subclass = agent['subclass']
                agent_data = df[(df['tracking_id'] == tracking_id) & (df['subclass'] == subclass)]

                # Predict trajectories
                path_predictions = predict_trajectories(model, agent_data, prediction_timestamps)
                if path_predictions is None:
                    continue  # Not enough data to make a prediction

                # Append predictions
                file_predictions.extend(path_predictions)

            # Save predictions for this file
            output_file = os.path.join(output_folder, f'Path_Prediction_Submission_Run_{run_number}.csv')
            if file_predictions:
                df_file_predictions = pd.concat(file_predictions, ignore_index=True)
                # Save to CSV with 9 decimal places for numerical columns
                df_file_predictions.to_csv(output_file, index=False, float_format='%.9f')
                print(f"Predicted trajectories saved to '{output_file}'")
                files_processed += 1
            else:
                # Output empty file with headers
                cols = ['Timestamps', 'tracking_ID', 'path_ID', 'subclass',
                        'x_center', 'y_center', 'z_center',
                        'x_length', 'y_length', 'z_length', 'confidence_score']
                df_empty = pd.DataFrame(columns=cols)
                df_empty.to_csv(output_file, index=False)
                print(f"No predictions were made for file '{filename}'. Empty file saved to '{output_file}'")
    print(f"\nTotal files processed: {files_processed}")

if __name__ == '__main__':
    main()
