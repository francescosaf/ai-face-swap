#!/usr/bin/env python3
import cv2, numpy as np

video_path = "/Users/francescosaf/Downloads/Comment PROMPT and I’ll send you the full prompt....Samay Raina, Ranveer Allahbadia, TRS, Latent.mp4"
photos = [
    "/Users/francescosaf/Downloads/foto1.jpeg",
    "/Users/francescosaf/Downloads/foto2.jpeg",
    "/Users/francescosaf/Downloads/foto3.jpeg"
]
output_path = "/Users/francescosaf/Downloads/final_result_3subjects.mp4"

cap = cv2.VideoCapture(video_path)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

images = [cv2.imread(p) for p in photos]
images = [cv2.resize(img, (frame_w//2, frame_h//2)) for img in images]
sizes = [img.shape[:2] for img in images]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_w, frame_h))

total_frames = int(fps * 15)
segment = total_frames // 3
count = 0
while count < total_frames:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    idx = min(count // segment, 2)
    img = images[idx]
    h, w = sizes[idx]
    x = (frame_w - w)//2
    y = (frame_h - h)//2
    frame[y:y+h, x:x+w] = img
    out.write(frame)
    count += 1

cap.release()
out.release()
print(f"VIDEO SALVATO: {output_path} ({count} frames, 15 sec, 3 soggetti)")
