import cv2
import numpy as np
import os


def bird_counter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, _ = cv2.connectedComponents(cleaned)
    return num_labels - 1


folder_path = "./bird_miniatures/"
counter = 0
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        bird_count = bird_counter(image_path)
        print(f"{filename}: {bird_count}")
        counter += bird_count

print(f"total: {counter}")
