# AI Face Swap — Video Deepfake Pipeline

Pipeline **locale** (senza API a pagamento) per sostituire i volti in un video,
usando **3 foto di riferimento** come sorgenti dei volti e un video di
riferimento come bersaglio. L'obiettivo è riprodurre in locale quello che
fanno servizi come Magic Hour, Remaker AI e Kling AI, usando solo codice
open-source: **Deep-Live-Cam** (ex Roop) + **InsightFace** + modelli **ONNX**.

> **Repository privato.** Contiene riferimenti a un video e a 3 foto di persone
> identificabili. Nessun media viene versionato (vedi `.gitignore`).

---

## Input

| Tipo | Percorso | Note |
|---|---|---|
| Video di riferimento | `~/Downloads/Comment PROMPT...Latent.mp4` | 9:16, 17.16 s |
| Foto 1 | `~/Downloads/foto1.jpeg` | 679x1024 |
| Foto 2 | `~/Downloads/foto2.jpeg` | 1536x2048 |
| Foto 3 | `~/Downloads/foto3.jpeg` | 685x1024 |
| Chiave API Magic Hour | `~/apikey.txt` | opzionale, solo per `magic_hour_call.py` |

I media stanno **fuori** dal repository e non vengono caricati su GitHub.

## Output

| File | Stato | Descrizione |
|---|---|---|
| `final_result_3subjects.mp4` | ✅ esiste | **NON è un face swap.** Video sorgente con le 3 foto statiche incollate al centro, cambiate ogni 5 s |
| `deep_ai.mp4` | ❌ mai prodotto | L'output atteso dal face swap reale |

## Stato reale del progetto

Il face-swap AI **non è mai stato eseguito**. La pipeline contiene due parti
scritte ma solo una usata, e quella usata non fa face swap.

### Cosa produce output oggi

`generate_final.py` legge i frame con OpenCV e assegna direttamente un
rettangolo di pixel:

```python
frame[y:y + h, x:x + w] = img   # generate_final.py:36
```

Nessun detector, nessun tracking, nessuna inferenza. Risultato: uno slideshow
di 3 foto statiche sopra il video originale.

Verifica: la differenza media tra due frame nella regione centrale (dove sta la
foto) è `0.153/255`, contro `4.679/255` nella regione di bordo dove il video
si muove normalmente. `0.15` è rumore di compressione H.264, non movimento.

### Cosa manca

1. **Nessun driver headless.** Deep-Live-Cam è pensato per la GUI Tkinter:
   ogni funzione di `modules/core.py` chiama `update_status()`. Va scritto un
   loop che usi direttamente `face_swapper.get_face_swapper()` e
   `face_enhancer.process_frame_v2()`.
2. **Nessuna mappatura multi-sorgente.** Deep-Live-Cam accetta una sola
   faccia sorgente. Serve codice esplicito per associare i 3 volti del video ai
   3 riferimenti.
3. **`gfpgan-1024.onnx` mancante.** È il modello che `face_enhancer.py` richiede
   (`MODEL_FILE = "gfpgan-1024.onnx"`). Il `GFPGANv1.4.pth` presente è un
   checkpoint PyTorch, inutilizzabile: torch non è installato e il codice è
   ONNX-only.
4. **`buffalo_l` scaricato in un percorso doppio** (`dlfolder/models/models/`),
   perché il codice passa `root='models'` e InsightFace aggiunge `models/`.

### Modelli verificati funzionanti

Test reali su macOS Intel (`x86_64`), venv Python 3.12.14,
onnxruntime 1.23.0, insightface 0.7.3:

| Provider | Init | Per-frame |
|---|---|---|
| CoreML | 17.1 s | **0.80 s** |
| CPU | 3.2 s | 1.39 s |

`inswapper_128.onnx` carica ed esegue correttamente. CoreML accelera 186 nodi su
226. Stima: 360 frame x 0.8 s ≈ **5 minuti** per un volto; 20-40 minuti con 3
soggetti e GFPGAN attivo. Lento ma fattibile.

### Bug noti

- `generate_final.py:18` — `cv2.resize` non preserva il rapporto aspettale:
  foto2 è 3:4 e viene schiacciata in 9:16, risultando deformata.
- `generate_final.py` — l'output è 15.00 s contro 17.16 s del sorgente:
  **2.16 s tagliati in silenzio**.
- `generate_final.py` — `cv2.imread` senza controllo su `None`: un percorso
  sbagliato produce `cv2.resize(None)` → crash.
- `faceswap_pipeline.py:48-49` — chiamate principali commentate;
  `swap_faces()` è un `shutil.copy`. Non è mai stato eseguito.
- `face_swap.py` — la via locale invoca `roop` (repository rinominato in
  Deep-Live-Cam, quel path non è più valido); le vie cloud fanno
  `raise NotImplementedError`. Codice morto.
- `magic_hour_call.py` — `SOURCE_IMG` non è definito → `NameError` all'esecuzione.

## Struttura del repository

```
AI face swap/
├── .gitignore
├── README.MD              # questo file
├── ARCHITECTURE.MD        # piano architetturale (non descrive il codice eseguito)
├── GOAL.MD                # obiettivo e note di stato
├── face_swap.py           # scaffolding multi-opzione — non funzionante
├── faceswap_pipeline.py   # pipeline placeholder — non eseguita
├── generate_final.py      # UNICO script eseguito: overlay OpenCV, non AI
└── magic_hour_call.py     # client API Magic Hour — non funzionante
```

`dlfolder/` è escluso: è un clone intatto di `hacksider/Deep-Live-Cam` 2.1.6
(commit `759e3f9`) **senza modifiche locali**, con dentro venv (1.9 GB) e modelli
(1.5 GB). GitHub rifiuta file oltre 100 MB, quindi committarlo bloccherebbe il
push. Per ricostruirlo:

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git dlfolder
cd dlfolder && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
# modelli: vedi dlfolder/models/instructions.txt
```

## Requisiti

- macOS Intel (`x86_64`) o Apple Silicon
- `ffmpeg` — su Intel: build statico da evermeet.cx (`brew` richiede
  Command Line Tools)
- Python 3.10+
- `onnxruntime`, `insightface`, `opencv-python` (nel venv di `dlfolder`)

## Note legali

- **`inswapper_128.onnx` è non-commercial, research use only** (InsightFace).
  Nessun uso commerciale senza licenza.
- Sostituire il volto di una persona reale e identificabile senza il suo consenso
  viola i diritti di immagine. Dal 2 agosto 2026 l'art. 50(4) dell'AI Act impone
  di dichiarare i deepfake in modo evidente all'utente.
- Per pubblicazione usare soggetti che abbiano autorizzato la sostituzione.

## Documentazione correlata

- `ARCHITECTURE.MD` — il piano originale. **Non corrisponde al codice eseguito**:
  descrive un output `deep_ai.mp4` via Deep-Live-Cam e una timeline a intervalli
  che lo script non implementa (usa terzi uguali).
- `GOAL.MD` — obiettivo originale e problemi incontrati. Contiene alcuni stati
  superati (ad esempio `buffalo_l`, che ora è scaricato).