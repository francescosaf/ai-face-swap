#!/usr/bin/env python3
"""Test script per headswap (non face-swap semplice).
Controlla: melting, warped face, alignment failure.
Output: /Users/francescosaf/Downloads/FaceSwapAlternative/
"""
import os, sys, cv2, numpy as np
sys.path.insert(0, "/Users/francescosaf/Workspace/PersonalPrj/AI face swap_temp")

OUT_DIR = "/Users/francescosaf/Downloads/FaceSwapAlternative"
os.makedirs(OUT_DIR, exist_ok=True)

# Import headswap modules
from head_swap import HeadSwapper, sane_head_mask, MIN_HEAD_RATIO, MAX_HEAD_RATIO
from face_parsing import FaceParser, FACE_CLASSES, HAIR_CLASSES, HIGH_FREQ_CLASSES

def test_head_melting(video_path="/Users/francescosaf/Downloads/deep_ai.mp4"):
    cap = cv2.VideoCapture(video_path)
    bad = 0
    for _ in range(30):
        ret, f = cap.read()
        if not ret: break
        # Melting proxy: se il centro faccia è troppo omogeneo o troppo variabile
        h, w = f.shape[:2]
        roi = f[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        std = float(np.std(roi))
        if std < 12 or std > 130:
            bad += 1
    cap.release()
    report = f"Head-Swap Melting suspects: {bad} frames\n"
    with open(os.path.join(OUT_DIR,"headswap_melting_report.txt"),"w") as fh:
        fh.write(report)
    print(report)

def test_head_warp(video_path="/Users/francescosaf/Downloads/deep_ai.mp4"):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for _ in range(30):
        ret, f = cap.read()
        if not ret: break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in faces:
            ratios.append(float(w)/max(h,1))
    cap.release()
    r = np.array(ratios) if ratios else np.array([1.0])
    mean_r = float(np.mean(r))
    std_r = float(np.std(r))
    status = "WARP ALERT" if mean_r < 0.75 or mean_r > 1.35 or std_r > 0.15 else "OK"
    report = f"Head-Swap Warp: mean_ratio={mean_r:.2f}, std={std_r:.2f} -> {status}\n"
    with open(os.path.join(OUT_DIR,"headswap_warp_report.txt"),"w") as fh:
        fh.write(report)
    print(report)

def test_head_alignment(video_path="/Users/francescosaf/Downloads/deep_ai.mp4"):
    # Usa sane_head_mask per verificare se il bbox della testa è plausibile
    # Questo test usa un frame campione (frame 10)
    cap = cv2.VideoCapture(video_path)
    for _ in range(10): cap.read()
    ret, f = cap.read()
    cap.release()
    if not ret:
        print("No frame for alignment test")
        return
    # Simula un bbox grande per testare sane_head_mask
    h, w = f.shape[:2]
    fb = (int(w*0.25), int(h*0.2), int(w*0.5), int(h*0.6))
    # Dummy labels per test
    labels = np.zeros((h,w), dtype=np.uint8)
    labels[int(h*0.3):int(h*0.7), int(w*0.35):int(w*0.65)] = 1
    kps = np.array([[w*0.4,h*0.3],[w*0.6,h*0.3],[w*0.5,h*0.5],[w*0.45,h*0.65],[w*0.55,h*0.65]], dtype=np.float32)
    mask, fallen = sane_head_mask(labels, fb, kps)
    report = f"Head-Swap Alignment (sane_head_mask): fallen_back={fallen}, mask_area={int(mask.sum())}, bbox_area={fb[2]*fb[3]}\n"
    with open(os.path.join(OUT_DIR,"headswap_alignment_report.txt"),"w") as fh:
        fh.write(report)
    print(report)

if __name__ == "__main__":
    print("=== HeadSwap Quality Tests ===")
    print("Branch: feature/headswap-test")
    print("Output dir:", OUT_DIR)
    test_head_melting()
    test_head_warp()
    test_head_alignment()
    print("=== Done ===")
