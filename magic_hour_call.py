#!/usr/bin/env python3
"""Magic Hour AI — Face Swap Video (API gratuita con crediti)
Registrati su: https://magichour.ai/developer (free keys, no card)
"""
import os, requests, time

API_KEY = open("/Users/francescosaf/apikey.txt").read().strip()
OUTPUT_DIR = "/Users/francescosaf/Downloads"
VIDEO_PATH = "/Users/francescosaf/Downloads/da0a219f-1cda-4ae1-8cbe-6dc5b7ef4538.mp4"
PHOTO_PATHS = [
    "/Users/francescosaf/Downloads/foto1.jpeg",
    "/Users/francescosaf/Downloads/foto2.jpeg",
    "/Users/francescosaf/Downloads/foto3.jpeg",
]
VIDEO_PATH = "/Users/francescosaf/Downloads/Comment PROMPT and I’ll send you the full prompt....Samay Raina, Ranveer Allahbadia, TRS, Latent.mp4"

headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Upload (o usa URL pubblico)
# Per semplicità, assumiamo file già accessibile o caricato tramite API upload
# 2. Crea job
url = "https://api.magichour.ai/api/v1/face-swap/video"
payload = {
    "image_file_path": SOURCE_IMG,  # deve essere URL pubblico o path caricato
    "video_file_path": VIDEO_PATH,
    "start_seconds": 0,
    "end_seconds": 15
}

print("[Magic Hour] Inviando job...")
r = requests.post(url, headers=headers, json=payload)
print(r.json())
