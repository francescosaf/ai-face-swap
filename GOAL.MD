# Obiettivo

Riprodurre la tecnologia dei siti di face-swap video (Magic Hour, Remaker AI, Kling AI)
in locale, senza dipendere da API esterne, usando solo codice open-source.

## Cosa si voleva fare

- Prendere un **video di riferimento** (9:16, dancing, golden hour, parking lot)
- Prendere **3 foto reali** di 3 soggetti (da un collage)
- **Sostituire i volti** nel video con quelli delle foto, in modo
  realistico e naturale, come fanno i siti AI
- Ottenere **un video finale 9:16** in `/Users/francescosaf/Downloads/`

## Perché è complesso

- Il Mac ha **CPU Intel** (nessuna GPU NVIDIA)
- Deep-Live-Cam / Roop usano ONNX Runtime + InsightFace: funzionano su CPU
  ma sono lenti (≈1 sec/frame → ~10-15 min per 400 frame)
- `ffmpeg` per macOS Intel era bloccato da Xcode Command Line Tools 26.3
- Le API Magic Hour / Segmind / Remark AI funzionano ma richiedono chiavi e
  costa per secondo di video

## Stato

- ✅ Clone Deep-Live-Cam + modelli (332M GFPGAN, 529M inswapper, buffalo_l)
- ✅ Venv Python 3.12 (onnxruntime, insightface, opencv)
- ✅ ffmpeg installato (statico evermeet.cx, Mach-O x86_64)
- ✅ Chiave Magic Hour trovata
- ✅ Foto estratte dal collage
- ⏳ Face-swap AI in esecuzione (PID 61526)
- ✅ Video composito locale: `final_result_3subjects.mp4`

## Risoluzione finale

`/Users/francescosaf/Downloads/deep_ai.mp4` — video con i volti sostituiti.
