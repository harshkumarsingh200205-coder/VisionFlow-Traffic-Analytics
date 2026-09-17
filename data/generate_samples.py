"""
generate_samples.py — Generates synthetic sample image and video files for test demonstration.
"""
import os
import cv2
import numpy as np

os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# 1. Generate Synthetic Traffic Scene Image
img1 = np.full((720, 1280, 3), (35, 40, 45), dtype=np.uint8)
# Road surface
cv2.rectangle(img1, (0, 350), (1280, 720), (55, 60, 65), -1)
# Road lane markings
for x in range(50, 1280, 160):
    cv2.rectangle(img1, (x, 520), (x + 80, 535), (220, 220, 220), -1)
# Vehicle 1: Blue Sedan
cv2.rectangle(img1, (200, 430), (420, 560), (180, 80, 40), -1)
cv2.putText(img1, "CAR", (280, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
# Vehicle 2: Green Bus
cv2.rectangle(img1, (650, 370), (950, 580), (50, 160, 60), -1)
cv2.putText(img1, "BUS", (760, 485), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

img_path = "data/input/sample_traffic_image.jpg"
cv2.imwrite(img_path, img1)
print(f"Generated {img_path}")

# 2. Generate Synthetic Moving Traffic Video Clip
video_path = "data/input/sample_traffic_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 360))

for i in range(80):
    frame = np.full((360, 640, 3), (40, 45, 50), dtype=np.uint8)
    cv2.rectangle(frame, (0, 180), (640, 360), (60, 65, 70), -1)
    # Moving vehicle across frames
    cx = int(20 + i * 7)
    cv2.rectangle(frame, (cx, 220), (cx + 110, 285), (6, 182, 212), -1)
    cv2.putText(frame, "CAR", (cx + 25, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    out.write(frame)

out.release()
print(f"Generated {video_path}")
