# ARCHITECTURE — AI Face-Swap Video Pipeline

## Repository

Tre entry point, condivisi da CLI, notebook e verifica. Tutta la logica sta
in questi file; `dlfolder/` è un clone stock di Deep-Live-Cam **non modificato**,
escluso da Git perché contiene venv e modelli.

| File | Ruolo |
|---|---|
| `headless_faceswap.py` | entry point: risolve gli input, sceglie l'hardware, sostituisce |
| `abbina_volti.py` | censimento: cluster + tabella frame → persona → foto |
| `video_face_roster.py` | analizzatore: conta le persone, produce provini e config |
| `verifica_swap.py` | misura se lo swap è avvenuto davvero |
| `diagnosi_difetti.py` | melting, warping, allineamento, bordo |
| `misura_flicker.py` | sfarfallio frame per frame rispetto all'originale |
| `verifica_headswap.py` | distingue head swap da face swap |
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
| GFPGAN | `gfpgan-1024.onnx` | enhancement: **da non usare**, vedi sotto |
| `buffalo_l` | cache InsightFace | detector + embedding |

InsightFace scarica `buffalo_l` da solo in `~/.insightface/models/`.

**GFPGAN è stato provato ed è la causa dello sfarfallio**: spingeva i dettagli
della pelle al 400% dell'originale, e quel contrasto in più sul video non
swappato si legge come flickering. Tutte le run di faceswap passano per
`--no-enhancer`; il percorso resta nel codice ma va considerato rotto.

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

## Layer 5b — Mappa frame → foto (faceswap multi-persona)

Con più persone, associare le foto **per persona** non basta: due algoritmi di
raggruppamento possono contare identità diverse, e `--map` finisce sugli
indici sbagliati con un output plausibile ma sulla persona errata. Il passo
affidabile è la mappa **per frame**, prodotta da `abbina_volti.py`.

```
video ── InsightFace ── tracciati ── Otsu ── cluster
                                              │
foto ── embedding ── somiglianze (solo informative)
                                              │
                              --assegna 1,1,2,0,0,0,0,1   ← deciso a mano
                                              │
                              match_frame_foto.csv
                                frame,persona,x1,y1,x2,y2,
                                foto_assegnata,foto_indice
                                              │
                    headless_faceswap.py --mappa-file ... CSV
```

Le somiglianze `sim_*.jpeg` vengono **riportate mai usate**: se il soggetto
della foto non è nel video sono rumore (0.0–0.2) e assegnare in automatico
produce abbinamenti senza senso. `foto_indice` è 0-based sull'ordine di
`--photo`, e `-1` lascia il volto originale.

**Scelta della riga**, per ogni volto rilevato in ogni frame
(`headless_faceswap.py`):

1. fra le righe del frame si prende l'IoU maggiore, valido se `≥ 0,30`;
2. se il frame non ha righe, si cercano i **6 frame adiacenti** e si prende la
   riga più vicina, con distanza fra i centri normalizzata sulla **larghezza
   del volto rilevato**: una larghezza è il limite, oltre è un'altra persona;
3. nessuna riga valida → il volto resta com'è e viene contato (`Altre: N`).

Il passo 2 è stato un bug vero: la normalizzazione sul lato maggiore del bbox
assorbiva 450 px di distanza per un volto alto 600 px, quindi la seconda
persona di un piano prendeva la foto del soggetto principale.

**Soglia del detector.** `get_many_faces` lavora a 0.50, ma i volti ai margini
e in moto escono con 0.26–0.47 e venivano scartati: il frame tornava
all'originale e quello dopo riprendeva, un lampeggio leggibile come flickering.
In `--mode faceswap` la soglia viene abbassata a **0.25 per tutti i frame**
avvolgendo `get_many_faces` (`_soglia_ridotta`): la mappa decide comunque chi
sostituire, quindi cambia solo quanti volti vengono *visti*, non quale foto
gli venga data. In `headswap` resta 0.50, dove la soglia entra nel
tracciamento. L'eventuale retry da 0.25 su chiamata vuota non basta: se a
0.50 viene trovato un volto marginale, il volto giusto resta fuori.

La scelta sta nel nostro script e non in `dlfolder/modules/face_analyser.py`:
`dlfolder/` è una repo separata con remote upstream, esclusa dal
versionamento, e una modifica lì andrebbe persa al prossimo rebuild.

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
---

## Trapianto dei capelli (head swap reale)

Il ricoloramento dei capelli non rende un head swap. La capigliatura della
foto viene allineata al volto del target e incollata.

Ordine delle trasformazioni, in `head_swap.py`:

1. `M_src` e `M_tgt` portano dal frame reale al template 112 di ArcFace
2. `A = inv(M_tgt) @ M_src` mappa la sorgente sul target
3. la maschera capelli e l'immagine sorgente vengono warpiate con `A`
4. la regione di arrivo è un'ellisse derivata dal bounding box del viso,
   tagliata sotto il mento
5. il volto è escluso con margine (ellisse 1.30 + dilatazione 7×7): i capelli
   che sfiorano gli occhi spostano i landmark
6. il colore viene riallineato all'illuminazione del target in spazio LAB
   (`_match_lab`), altrimenti il trapianto si legge come un collage
7. l'alpha è sfumato 3.0 su due lati

**L'ordine delle matrici è critico e va verificato, non supposto.**
`M_src @ p_src = M_tgt @ p_tgt` implica `p_tgt = inv(M_tgt) @ M_src @ p_src`.
Scrivendo `M_tgt @ inv(M_src)` si ottiene una scala di 1.92 invece di 0.515:
i capelli si espandono di 2.7x invece di adattarsi al volto. Il sintomo
(maschera grande, trapianto gonfio) è identico a quello della prima versione
a foto statica, quindi la verifica va fatta sul determinante della
trasformazione, non a occhio.

La maschera dei capelli **non** va intersecata con quella del target: quella
descrive la capigliatura già presente nel video. Intersecandole si perdeva il
92% dei capelli (52855 px allineati → 4042 px utili).

## Correzioni di plumbing

- `--quality` sovrascriveva gli override espliciti di `--temporal`,
  `--blend` e `--face-scale`, perché `__init__` riapplicava i valori del preset
  dopo aver ricevuto i parametri. Il commento nel CLI diceva il contrario.
  I default dei parametri sono ora `None`, così "non passato" si distingue da
  "passato" e l'override vince sul preset.
- `q` era referenziato fuori dal `if` che lo definiva: senza `--quality` il
  modulo andava in `NameError` all'istanziazione.

## Temporal e geometria

Il peso del temporal ha un costo misurabile in allineamento, non solo in
stabilità:

| temporal | proporzioni bocca/occhi | spostamento landmark | flickering |
|---|---|---|---|
| 0.26 | 4.06% | 4.68 px | 0.073 |
| 0.44 | 5.72% | 5.58 px | 0.077 |

Lo smoothing temporale introduce ritardo e sposta i tratti. I preset sono
tarati su `fast` 0.26, `high` 0.30, `maximum` 0.38.

## Maschera semantica

La maschera del volto è *ellipse-first*: la semantica di BiSeNet interseca
l'ellisse invece di allargarla. La versione precedente faceva
`or(ellisse, and(sem, ellisse))`, che è un'ellisse dilatata: allargare la
maschera oltre i limiti del volto fa uscire il trapianto dal viso ed è la
causa del melting.
