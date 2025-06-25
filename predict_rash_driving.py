import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser

# -------- CONFIGURATION -------- #
MODEL_PATH = 'rash_driving_model.h5'
RASH_THRESHOLD = 0.30

# Pixel to real-world conversion (fine-tune this)
# Try adjusting between 0.012 and 0.018 for best accuracy
METERS_PER_PIXEL = 0.013 

# Approximate real widths of objects (in meters)
OBJECT_WIDTHS = {
    'car': 1.8, 
    'motorcycle': 0.8, 
    'bus': 2.5, 
    'truck': 2.5, 
    'person': 0.5 
}
# -------------------------------- #

model = load_model(MODEL_PATH)

def preprocess_features(csv_file):
    data = pd.read_csv(csv_file)
    print(f"[INFO] Loaded {csv_file}")

    data[['Frame Index', 'Time (s)', 'Center X', 'Center Y', 'Width', 'Height']] = \
        data[['Frame Index', 'Time (s)', 'Center X', 'Center Y', 'Width', 'Height']].apply(pd.to_numeric, errors='coerce')

    if data.isnull().any().any():
        print(f"[WARN] NaN values detected, replacing with 0.")
        data = data.fillna(0)

    features = data[['Center X', 'Center Y', 'Width', 'Height']].values

    if features.shape[0] == 0:
        print(f"[WARN] Empty features in {csv_file}.")
        return None, data

    sequence_length = 30
    total_frames = features.shape[0]

    if total_frames < sequence_length:
        padding = np.zeros((sequence_length - total_frames, features.shape[1]))
        features = np.vstack([features, padding])

    if features.shape[0] % sequence_length != 0:
        padding = np.zeros((sequence_length - features.shape[0] % sequence_length, features.shape[1]))
        features = np.vstack([features, padding])

    features = features.reshape(-1, sequence_length, features.shape[1])
    return features, data

def predict_rash_driving(features):
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def estimate_meters_per_pixel(obj_width_pixels, class_name):
    """Dynamically estimate meters per pixel based on object size."""
    if class_name in OBJECT_WIDTHS:
        real_width = OBJECT_WIDTHS[class_name]
        return real_width / obj_width_pixels if obj_width_pixels > 0 else METERS_PER_PIXEL
    return METERS_PER_PIXEL  # Default fallback

def calculate_speed_kmph(data):
    grouped = data.groupby('Track ID')

    max_speed = 0
    rash_object = None

    for track_id, track_data in grouped:
        track_data = track_data.sort_values(by='Frame Index')

        speeds = []
        for i in range(1, len(track_data)):
            x1, y1, t1, width1, class1 = track_data.iloc[i-1][['Center X', 'Center Y', 'Time (s)', 'Width', 'Class Name']]
            x2, y2, t2, width2, class2 = track_data.iloc[i][['Center X', 'Center Y', 'Time (s)', 'Width', 'Class Name']]

            # Estimate meters per pixel dynamically
            mpp1 = estimate_meters_per_pixel(width1, class1)
            mpp2 = estimate_meters_per_pixel(width2, class2)
            meters_per_pixel = (mpp1 + mpp2) / 2  # Smooth interpolation

            # Calculate real-world speed
            distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distance_meters = distance_pixels * meters_per_pixel
            time_seconds = t2 - t1

            if time_seconds > 0:
                speed_mps = distance_meters / time_seconds
                speeds.append(speed_mps)

        if speeds:
            avg_speed_mps = np.mean(speeds)
            avg_speed_kmph = avg_speed_mps * 3.6  # Convert to km/h

            if avg_speed_kmph > max_speed:
                max_speed = avg_speed_kmph
                rash_object = track_data.iloc[0]

    return rash_object, max_speed

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_rash_driving.py <csv_file_path>")
        sys.exit(1)

    csv_file = sys.argv[1]

    features, original_data = preprocess_features(csv_file)
    if features is None:
        print("[INFO] No features to predict.")
        return

    predicted_class = predict_rash_driving(features)

    rash_frames = np.sum(predicted_class == 1)
    total_frames = len(predicted_class)

    print(f"[RESULT] ➔ {os.path.basename(csv_file)}")
    print(f"  ➔ Total sequences: {total_frames}")
    print(f"  ➔ Rash frames: {rash_frames} ({(rash_frames / total_frames) * 100:.2f}%)")

    if rash_frames / total_frames > RASH_THRESHOLD:
        print("  ➔ FINAL PREDICTION:  Rash Driving Detected!")

        # Calculate real speed
        rash_object, speed_kmph = calculate_speed_kmph(original_data)

        if rash_object is not None:
            print(f"\n[DETAILS] Rash Object:")
            print(f"  ➔ Class: {rash_object['Class Name']}")
            print(f"  ➔ Speed: {speed_kmph:.2f} km/h")
        else:
            print("\n[DETAILS] No moving object detected.")
        rash_html_path = r"C:\Users\kapil\Downloads\myvenv\rash\rash.html"
        webbrowser.open(f"file:///{rash_html_path}")
        
        
    else:
        print("  ➔ FINAL PREDICTION:  Safe Driving")
        safe_html_path = r"C:\Users\kapil\Downloads\myvenv\rash\safe.html"
        webbrowser.open(f"file:///{safe_html_path}")

if __name__ == "__main__":
    main()
