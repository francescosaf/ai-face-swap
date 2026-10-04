# ARCHITECTURE — AI Face-Swap Video Pipeline

## Repository

Due soli entry point, condivisi da CLI, notebook e verifica. Tutta la logica sta
in questi file; `dlfolder/` è un clone stock di Deep-Live-Cam **non modificato**,
escluso da Git perché contiene venv e modelli.

| File | Ruolo |
|---|---|
| `headless_faceswap.py` | entry point: risolve gli input, sceglie l'hardware, sostituisce |
| `video_face_roster.py` | analizzatore: conta le persone, produce provini e config |
| `verifica_swap.py` | misura se lo swap è avvenuto davvero |
| `face_parsing.py` | wrapper BiSeNet: segmentazione della testa in 19 classi |
| `head_swap.py` | composizione dell'head swap |
| `TestFaceSwap.ipynb` | Colab: esegue gli script sopra con `subprocess` |
| `dlfolder/` | Deep-Live-Cam 2.1.6, invariato |

## Layer 1 — Ingressi

- video: qualunque formato leggibile da ffmpeg
- N foto di riferimento, o un `swap_config.json` prodotto dall'analizzatore
- `--prompt`: opzionale, solo documentale

Le foto non vengono modificate. Vengono solo analizzate una volta e messe in
cache per il video.

## Layer 2 — Modelli

| Modello | File | Ruolo |
|---|---|---|
| `inswapper_128` | `inswapper_128.onnx` (529 MB) | sostituisce il volto interno |
| `inswapper_128_fp16` | scaricato solo su CUDA | variante half precision |
| BiSeNet ResNet18 | `bisenet_resnet18.onnx` (51 MB, MIT) | segmenta la testa in 19 classi |
| GFPGAN | `gfpgan-1024.onnx` | **mai testato**, vedi GOAL.MD |
| `buffalo_l` | cache InsightFace | detector + embedding |

InsightFace scarica `buffalo_l` da solo in `~/.insightface/models/`.

Il parsing BiSeNet usa le 19 classi di CelebAMask-HQ:

```
0 background   1 skin      2 l_brow    3 r_brow    4 l_eye     5 r_eye
6 eyeglass     7 l_ear     8 r_ear     9 earring   10 nose     11 mouth
12 u_lip      13 l_lip    14 neck     15 necklace 16 cloth    17 hair
18 hat
```

## Layer 3 — Selezione hardware

Ordine di preferenza: **CUDA → CoreML → CPU**.

Su Colab, se CUDA non è ancora registrato in ONNX Runtime, viene installato
`onnxruntime-gpu` automaticamente. Senza questo, il controllo availability
sceglieva la CPU su Colab anche con GPU presente.

## Layer 4 — Rilevamento e raggruppamento

1. **Tracciati temporali**: i volti fra frame consecutivi sono collegati se i
   bounding box hanno IoU > 0.25. L'embedding di ogni tracciato è la media dei
   suoi membri.
2. **Clustering agglomerativo** con linkage average su distanza coseno.
3. **Soglia di Otsu** sulle altezze di merge, invece di una costante fissa.

Il passaggio 1 è quello che rende il risultato usable: senza la media, lo
stesso soggetto in pose diverse si frammenta. Sul video di prova:

| Metodo | Identità trovate |
|---|---|
| greedy, soglia fissa 0.55 | 79 |
| Otsu su embedding grezze | 26 |
| Otsu su embedding mediate per tracciato | **4** |

La soglia usata viene stampata a console e il raggruppamento è ispezionabile.

## Layer 5 — Assegnazione foto ↔ persona

Non è deducibile. La distanza embedding fra una foto e un volto del video è
0.90+, il che significa solo "sono persone diverse". Quindi:

- le persone sono numerate **in ordine di prima apparizione** nel video, non
  per frequenza: `persona 1` è la prima che si vede
- automatico: le foto vengono abbinate in quell'ordine, con avviso a console
- manuale: `--map 0,1` (indice della foto per ogni persona, in ordine di
  persona)
- ispezione: `persona_N_provino.png` + `persone_trovate.txt`

Nessun soggetto è codificato nel sorgente: la numerazione e l'abbinamento
derivano dai dati del video.

Il video di prova ha una peculiarità: il sorgente è stato **generato con un'altra
applicazione AI**, e i volti sintetici hanno embedding più instabili di quelli
reali. È il motivo per cui la media sui tracciati serve più del solito.

## Layer 6 — Head swap

Ordine delle operazioni in `head_swap.py`, per frame e per volto:

1. `inswapper` sostituisce il volto interno → identità dalla foto
2. BiSeNet segmenta foto e video → capelli, orecchie, cappello separati da collo e abbigliamento
3. **nessun warping della testa**: la geometria, i capelli e lo sfondo restano
   quelli del target. I capelli e la pelle vengono solo ricolorati, riscalando
   e traslando i pixel sulla struttura del video, con miscelazione su
   maschera sfumata (una maschera binaria lasciava un bordo netto visibile)
4. **ricoloratura**: il rapporto dei colori della pelle allinea la sorgente all'illuminazione del target
5. **composizione**: `out = out·(1−α) + testa_warpiata·α`
6. **protezione del collo**: `α` forzato a zero sul collo
7. **Poisson blending** (seamlessClone MIXED) dopo inswapper per integrare
   il volto e ridurre glitch sul naso
8. sfumatura del bordo e passaggio a GFPGAN (se attivo)
9. smorzamento temporale (default 0.42)

Due dettagli che sono stati bug reali, entrambi trovati con le metriche:

- La ricoloratura applicava il rapporto a un array `ones_like(out)`, quindi
  **all'intero frame**: lo sfondo si sporcava di 53/255. Ora il rapporto si
  applica solo ai pixel della testa di riferimento → 0.83/255.
- La protezione del collo veniva sottratta dalla maschera **prima** della
  sfumatura, quindi il blur del bordo inferiore della testa ci rientrava:
  32.6/255. Applicandola **dopo** → 1.06/255.

`swap_face` scrive in place sul percorso GPU ma restituisce una copia su quello
CPU, quindi il valore di ritorno viene sempre assegnato.

## Layer 7 — Output

`ffmpeg` con `-framerate` dall'originale, `-c:a aac`, `-movflags +faststart`.
L'audio viene mappato dalla sorgente con `-map 1:a:0?`.

Su 409 frame a 24 fps il video dura 17.04 s contro i 17.16 s del formato
sorgente: **non è un troncamento**, i 409 frame ci sono tutti. È l'audio che
traccia un filo più lungo e ffmpeg chiude sul primo stream che finisce.

## Flusso dati

```
video ──ffprobe──> dimensioni, fps, durata, frame
     │
     ├──ffmpeg──> frames/%06d.png (fps_mode passthrough)
     │                │
     │                ├── InsightFace ──> tracciati ──> clustering ──> identita'
     │                │
     │                └── per frame e per volto:
     │                      BiSeNet   ──> maschera testa target
     │                      BiSeNet   ──> maschera testa foto (una volta, in cache)
     │                      landmark  ──> similar-transform
     │                      warping + ricoloratura + protezione collo
     │                      inswapper ──> volto interno
     │                      (GFPGAN)  ──> enhancement
     │                            │
     └────────────────────────────┴──ffmpeg──> output + audio
```

## Struttura a runtime

```
dlfolder/
├── venv/                    Python 3.12, non versionato
└── models/
    ├── inswapper_128.onnx         529 MB
    └── bisenet_resnet18.onnx       51 MB
```

I frame temporanei stanno in un `tempfile.mkdtemp()` e vengono rimossi alla
fine, salvo `--keep-frames`.