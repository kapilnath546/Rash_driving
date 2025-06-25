import os
import pandas as pd
import numpy as np
import cv2
import subprocess
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------- CONFIGURATION -------- #
VIDEO_INPUT_PATH = 'input/'
FEATURE_OUTPUT_PATH = 'features/'
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.3
SIMILARITY_FRAMES_CHECK = 15
SIMILARITY_THRESHOLD = 0.98
# -------------------------------- #

# COCO classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

os.makedirs(FEATURE_OUTPUT_PATH, exist_ok=True)
yolo_model = YOLO(YOLO_MODEL_PATH)
tracker = DeepSort(max_age=30)

def frames_are_similar(frame1, frame2, tolerance=15):
    if len(frame1) != len(frame2):
        return False
    matched = 0
    for det1, det2 in zip(frame1, frame2):
        bbox1, _, class_id1 = det1
        bbox2, _, class_id2 = det2
        if class_id1 != class_id2:
            continue
        diff = np.abs(np.array(bbox1) - np.array(bbox2))
        if np.all(diff < tolerance):
            matched += 1
    similarity = matched / max(1, len(frame1))
    return similarity > SIMILARITY_THRESHOLD

def process_video(video_path, output_path_csv):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Processing {video_path} at {fps:.2f} FPS")

    frames_detections = []
    frame_idx = 0
    early_skip = False
    last_positions = {}  # track_id : (center_x, center_y, time)

    with open(output_path_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Index', 'Time (s)', 'Track ID', 'Class Name', 'Center X', 'Center Y', 'Width', 'Height', 'Speed'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps

            results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                detections.append([[x, y, w, h], conf, class_id])

            if frame_idx <= SIMILARITY_FRAMES_CHECK:
                frames_detections.append(detections)

            if frame_idx == SIMILARITY_FRAMES_CHECK:
                all_similar = True
                for i in range(1, len(frames_detections)):
                    if not frames_are_similar(frames_detections[0], frames_detections[i]):
                        all_similar = False
                        break
                if all_similar:
                    print(f"[INFO] Early exit: {video_path} - Static early frames detected!")
                    early_skip = True
                    break

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()

                center_x = (l + r) / 2
                center_y = (t + b) / 2
                width = r - l
                height = b - t

                class_id = track.det_class if hasattr(track, 'det_class') else None
                class_name = COCO_CLASSES[class_id] if class_id is not None and 0 <= class_id < len(COCO_CLASSES) else "unknown"

                # Speed calculation (pixels per second)
                speed = 0
                if track_id in last_positions:
                    last_x, last_y, last_time = last_positions[track_id]
                    dist = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)
                    time_diff = current_time - last_time
                    if time_diff > 0:
                        speed = dist / time_diff  # pixels/sec

                last_positions[track_id] = (center_x, center_y, current_time)

                writer.writerow([frame_idx, current_time, track_id, class_name, center_x, center_y, width, height, speed])

    cap.release()

    if early_skip:
        if os.path.exists(output_path_csv):
            os.remove(output_path_csv)
        print(f"[INFO] Skipped {video_path} due to static early frames.")
    else:
        print(f"[INFO] Features saved to {output_path_csv}")
        print(f"[INFO] Running prediction for {output_path_csv}...")
        subprocess.run(['python', 'predict_rash_driving.py', output_path_csv])

def find_unprocessed_videos():
    video_files = [f for f in os.listdir(VIDEO_INPUT_PATH) if f.endswith(('.mp4', '.avi', '.mov'))]
    processed_files = [f for f in os.listdir(FEATURE_OUTPUT_PATH) if f.endswith('_features.csv')]
    processed_basenames = set(os.path.splitext(f)[0].replace('_features', '') for f in processed_files)

    unprocessed_videos = []
    for video in video_files:
        video_name_no_ext = os.path.splitext(video)[0]
        if video_name_no_ext not in processed_basenames:
            unprocessed_videos.append(video)

    return unprocessed_videos

def main():
    unprocessed_videos = find_unprocessed_videos()
    if not unprocessed_videos:
        print("[INFO] No new videos to process.")
        return

    print(f"[INFO] Found {len(unprocessed_videos)} unprocessed video(s):")
    for video in unprocessed_videos:
        print(f" - {video}")

    for video in unprocessed_videos:
        video_path = os.path.join(VIDEO_INPUT_PATH, video)
        feature_path_csv = os.path.join(FEATURE_OUTPUT_PATH, f"{os.path.splitext(video)[0]}_features.csv")
        process_video(video_path, feature_path_csv)

if __name__ == "__main__":
    main()
