import cv2
import json
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import Counter


class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.class_names = self.model.names

    def detect_image(self, image_path, confidence_threshold=0.5):
        results = self.model(image_path, conf=confidence_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls.item()),
                    "class_name": self.class_names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "bbox": box.xyxy[0].tolist(),
                }
                detections.append(detection)

        return detections, results[0].plot()

    def detect_video(self, video_path, confidence_threshold=0.5):
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        processed_frames = []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=confidence_threshold)
            frame_detections_frame = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "frame_number": frame_count,
                        "class_id": int(box.cls.item()),
                        "class_name": self.class_names[int(box.cls.item())],
                        "confidence": float(box.conf.item()),
                        "bbox": box.xyxy[0].tolist(),
                    }
                    frame_detections_frame.append(detection)

            frame_detections.append(
                {"frame_number": frame_count, "detections": frame_detections_frame}
            )

            processed_frames.append(results[0].plot())
            frame_count += 1

        cap.release()
        return frame_detections, processed_frames

    def save_detections_to_json(self, detections, output_path):
        with open(output_path, "w") as f:
            json.dump(detections, f, indent=2)

    def save_annotated_image(self, image, output_path):
        cv2.imwrite(output_path, image)

    def save_annotated_video(self, frames, output_path, fps=30):
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

    def get_class_statistics(self, video_detections):
        all_detections = []
        for frame_data in video_detections:
            all_detections.extend(frame_data["detections"])

        class_counts = Counter([det["class_name"] for det in all_detections])
        return dict(class_counts)


def main():
    output_dir = Path("zad1_res")
    output_dir.mkdir(exist_ok=True)

    detector = YOLODetector()

    image_path = "./data/office_yolo.png"
    video_path = "./data/office_yolo.mp4"

    confidence_thresholds = [0.1, 0.3, 0.5, 0.7]

    print("Przetwarzanie zdjęcia...")
    for conf in confidence_thresholds:
        print(f"  Confidence threshold: {conf}")

        detections, annotated_image = detector.detect_image(image_path, conf)

        json_path = output_dir / f"detections_image_conf_{conf:.1f}.json"
        detector.save_detections_to_json(detections, json_path)

        image_output_path = output_dir / f"annotated_image_conf_{conf:.1f}.png"
        detector.save_annotated_image(annotated_image, image_output_path)

        print(f"    Znaleziono {len(detections)} detekcji")

    print("\nPrzetwarzanie wideo...")
    for conf in confidence_thresholds:
        print(f"  Confidence threshold: {conf}")

        video_detections, processed_frames = detector.detect_video(video_path, conf)

        json_path = output_dir / f"detections_video_conf_{conf:.1f}.json"
        detector.save_detections_to_json(video_detections, json_path)

        video_output_path = output_dir / f"annotated_video_conf_{conf:.1f}.mp4"
        detector.save_annotated_video(processed_frames, video_output_path)

        stats = detector.get_class_statistics(video_detections)
        stats_path = output_dir / f"video_statistics_conf_{conf:.1f}.json"
        detector.save_detections_to_json(stats, stats_path)

        total_detections = sum(len(frame["detections"]) for frame in video_detections)
        print(f"    Łącznie detekcji: {total_detections}")
        print(f"    Statystyki klas: {stats}")


if __name__ == "__main__":
    main()
