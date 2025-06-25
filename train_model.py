import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# -------- CONFIGURATION -------- #
TRAINING_DATA_PATH = 'train/'  # Folder containing the labeled training data (CSV files inside 'safe/' and 'rash/')
SEQ_LENGTH = 30  # Sequence length for LSTM (number of frames per sample)
# -------------------------------- #

# Load and preprocess the data
def load_data():
    all_data = []
    all_labels = []

    for folder in ['safe', 'rash']:
        folder_path = os.path.join(TRAINING_DATA_PATH, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder_path, filename)
                data = pd.read_csv(filepath)

                # Extract features: 'Center X', 'Center Y', 'Width', 'Height'
                features = data[['Center X', 'Center Y', 'Width', 'Height']].values

                # For folder-based labeling:
                # 0 for safe, 1 for rash
                sequence_label_value = 0 if folder == 'safe' else 1

                # Create sequences from data with a sliding window
                for i in range(len(features) - SEQ_LENGTH):
                    sequence = features[i:i + SEQ_LENGTH]
                    all_data.append(sequence)
                    all_labels.append(sequence_label_value)

    return np.array(all_data), np.array(all_labels)

# Prepare data for LSTM
def preprocess_data():
    X, y = load_data()

    # Convert labels to categorical (binary classification: safe vs rash)
    y = to_categorical(y, num_classes=2)

    return X, y

# Define and train the LSTM model
def train_model():
    X, y = preprocess_data()

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Output layer for binary classification
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save('rash_driving_model.h5')
    print("[INFO] Model trained and saved!")

if __name__ == "__main__":
    train_model()
