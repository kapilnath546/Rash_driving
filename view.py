import numpy as np
import cv2
import os
import csv

# Load features from the .npy file
def load_features(file_path):
    features = np.load(file_path, allow_pickle=True)
    return features

# Display the first few frames of features
def display_features(features):
    print("First 5 frames of features:")
    for frame_features in features[:5]:  # Display the first 5 frames' features
        print(frame_features)

# Visualize bounding boxes on a frame
def visualize_frame(frame, features):
    if len(features) == 0:
        print("[INFO] No features to visualize in this frame.")
        return

    for feature in features:
        frame_idx, current_time, track_id, class_name, center_x, center_y, width, height = feature
        x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
        x2, y2 = int(center_x + width / 2), int(center_y + height / 2)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Frame with Bounding Boxes", frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Save features to a CSV file
def save_features_to_csv(features, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Index', 'Time (s)', 'Track ID', 'Class Name', 'Center X', 'Center Y', 'Width', 'Height'])

        for frame in features:
            for feature in frame:
                writer.writerow(feature)
    print(f"Features saved to {output_csv_path}")

# Main function
def main():
    # Get the list of .npy files in the 'features' folder
    features_folder = 'features/'
    features_files = [f for f in os.listdir(features_folder) if f.endswith('.npy')]

    if not features_files:
        print(f"No .npy files found in the {features_folder} directory.")
        return

    # Process each .npy file in the folder
    for features_file in features_files:
        features_file_path = os.path.join(features_folder, features_file)
        print(f"Loading features from {features_file_path}")

        # Load the features
        features = load_features(features_file_path)

        # Display the features (first 5 frames)
        display_features(features)

        # Extract the video filename from the .npy filename (e.g., from "youtube_pvDwr9cbbxU_638x360_h264_features.npy" to "youtube_pvDwr9cbbxU_638x360_h264.mp4")
        video_file_path = features_file_path.replace('.npy', '.mp4')  # Replace .npy with .mp4

        # Check if the video file exists
        if not os.path.exists(video_file_path):
            print(f"[ERROR] Video file not found: {video_file_path}")
            continue

        # Open the video file to visualize frames
        cap = cv2.VideoCapture(video_file_path)
        ret, frame = cap.read()
        if ret:
            visualize_frame(frame, features[0])  # Visualize features of the first frame
        else:
            print("[ERROR] Could not load video frame for visualization.")

        # Optional: Save the features to a CSV file
        output_csv_path = os.path.join(features_folder, f"{os.path.splitext(features_file)[0]}_features.csv")
        save_features_to_csv(features, output_csv_path)

if __name__ == "__main__":
    main()
