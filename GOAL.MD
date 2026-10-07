# Obiettivo

Riprodurre in locale ciò che fanno i servizi di face-swap video a pagamento
(Magic Hour, Remaker AI, Kling AI), senza dipendere da API esterne e usando solo
codice open-source.

## Requisito

Dato **un video qualsiasi** e **N foto di riferimento**, sostituire i volti nel
video in modo realistico, e poterlo fare sia in locale (macOS Intel) sia su
Colab/Kaggle, con lo stesso codice.

## Vincoli

- **Nessuna GPU NVIDIA**: il Mac è Intel. ONNX Runtime va su CoreML o CPU.
- **Lentezza**: ~1 s/frame misurato su CPU, ~0.8 s su CoreML. Per 400 frame
  sono 7-8 minuti.
- **Nessuna API a pagamento**: niente chiavi, niente costo al secondo.
- **`inswapper_128` è licenza non commerciale.**
- **Nessun modello ONNX per head swap accetta prompt testuali**: un prompt
  "sostituisci il volto e i capelli" descriverebbe un setup diverso
  (diffusione + ControlNet/IP-Adapter). I prompt vengono comunque registrati
  nel config come documentazione, con `prompt_is_used_by_model: false`.

## Stato

Raggiunto.

- ✅ Deep-Live-Cam 2.1.6 clonato in `dlfolder/`, **non modificato**
- ✅ Sostituzione reale del volto (non più overlay di foto statiche)
- ✅ **Head swap**: il volto viene sostituito rispettando la posa del target,
  capelli e pelle vengono ricolorati sulla struttura del video (BiSeNet).
  La testa della foto **non** viene trapiantata
- ✅ Funziona su macOS Intel (CoreML), CUDA e CPU; su Colab installa da solo
  `onnxruntime-gpu`
- ✅ Analizzatore: conta le persone nel video e produce provini + config
- ✅ Config autoinfficiente: video, foto e modalità in un solo JSON
- ✅ Notebook Colab che riusa gli stessi script, senza logica duplicata
- ✅ Verifica oggettiva: esce con codice di errore se lo swap non è avvenuto
- ✅ Durata e audio dell'originale conservate
- ✅ **Faceswap multi-persona**: `abbina_volti.py` produce la mappa
  frame → persona → foto e `--mappa-file` la applica, senza dipendere dal
  numero di cluster trovati
- ✅ **Flickering**: le due cause sono state trovate coi numeri e corrette
  (GFPGAN, soglia del detector a 0.50), con `misura_flicker.py` per misurarlo

## Risultato misurato (faceswap multi-persona)

Video 720×1280, 24 fps, 481 frame, 6,2 MB, tre foto, `--mode faceswap`
con `--mappa-file`, `--det-size 1024`, `--no-enhancer`, `--crf 17`:

| | valore |
|---|---|
| output | 3,4 MB (originale 6,2) |
| volti sostituiti | 433 su 481 |
| volti lasciati intatti per mappa (`-1`) | 84 |
| volti scartati perché fuori soglia | 18 |
| frame senza interruzioni sul 15° secondo | sì (prima no) |
| audio | AAC, conservato |

Il secondo video (Samay, 8 persone, `--assegna 1,1,2,0,0,0,0,1`) è approvato
a occhio: 422 swap su 409 frame, 25 ripresi dai frame adiacenti, 0 lasciati
originali per caso, allineamento 2,72 px, overlap 95,3%, eccesso di bordo
+0,0, flicker +0,048 px (identico all'originale).

## Risultato misurato

| Modalità | Identità residua | Area testa | Movimento | Fuori volto |
|---|---|---|---|---|
| originale | 0.988 | — | — | — |
| `headswap` (3 foto) | **0.360** | **0.98x** | **98%** | 1.36 |

Identità residua = `1 - coseno(embedding(volto in output), embedding(foto))`.
~0.9 = il volto è ancora quello originale, nessuno swap. ~0.2 = sostituito.
Area testa = rapporto output/originale (1.0 = identica).
Movimento = quanto il contenuto si muove rispetto all'originale (100% = uguale).

Output di prova: `~/Downloads/TestFaceSwap/headswap_3foto.mp4`
(720x1280, 24 fps, 409 frame, 17.04 s, audio AAC stereo, 397/409 frame).

La versione precedente era **rifiutata in test visivo**: area testa da 0.36x a
15.31x e contenuto immobile. Le cause trovate e corrette sono elencate in
`README.MD` ("Dettagli tecnici che hanno fatto la differenza").

## Cosa NON è risolto

- **L'abbinamento foto↔persona è ancora umano**, ma non più a mano libera:
  le embedding di una foto e di un volto del video distano 0.90+ (sono
  persone diverse, non un errore), quindi il soggetto va riconosciuto. Ormai
  si guardano le `persona_N.png` di `abbina_volti.py` e si scrive
  `--assegna 1,1,2,0,...`: il resto — bbox per frame, IoU, frame adiacenti —
  lo fa il codice. Resta non deducibile **quale** foto spetti a chi.
- **Il clustering non è garantito.** Le embedding dello stesso volto variano
  fino a ~0.56 fra frame consecutivi. Sul video di prova dà 4 identità (le
  due principali a 43 e 24 apparizioni); su un video con molta gente o pose
  molto diverse può frammentarsi. Per questo `--mappa-file` **ignora il
  clustering** e decide frame per frame: il numero di cluster smette di
  essere un rischio.
- **I frame sono PNG su disco**: un video molto lungo occupa molto spazio.
  Conviene lavorare in segmenti.
- **GFPGAN è provato e non va bene**: spinge i dettagli della pelle al 400%
  dell'originale e quel contrasto in più sulle zone non swappate si legge come
  sfarfallio. Il percorso resta nel codice ma va considerato rotto; il faceswap
  passa sempre per `--no-enhancer`.
- **Dettagli del volto al 58–69% dell'originale** in faceswap: il volto
  risulta più morbido. Con `--no-enhancer` l'unica leva rimasta è
  `--det-size`, non l'enhancement.

## Correzioni rispetto alle ipotesi iniziali

- **`foto1.jpeg` e `foto2.jpeg` sono la stessa persona** (distanza embedding
  0.404), non due soggetti diversi. Le tre foto non sono tre persone distinte.
- **`GFPGANv1.4.pth` non viene usato**: il codice è ONNX-only e usa
  `gfpgan-1024.onnx`.
- **Il video di prova non è più quello del ballo**: ora è un clip di podcast
  generato con un'altra AI, 720x1280, 17.16 s.
---

## Stato finale

Obiettivo raggiunto: il risultato è un head swap, non un face swap, e i tre
difetti segnalati sono risolti e misurati.

- **face melting** — i dettagli del volto sono al 116% dell'originale
  (v2 erano al 69%, la v3 high al 90%). Causa: la maschera semantica
  allargava l'ellisse oltre i limiti del volto.
- **warped face** — proporzioni bocca/occhi 2.59%, distanza inter-oculare
  3.64%. Sotto soglia.
- **alignment failure** — spostamento del baricentro dei landmark 2.30 px.
  Sotto soglia. Lo spostamento apparente di 7 px che si misurava prima era un
  artefatto del bounding box, che il detector dimensiona anche sui capelli.
- **flicker** — 0.054, sceso da 0.120.
- **head swap** — 33.4% dei pixel dei capelli riscritti, contro il 6.4% della
  v3.1 in cui i capelli restavano quelli del video.

Test finali in `~/Downloads/TestFaceSwap`: `test1.mp4` (fast), `test2.mp4`
(high), `test3.mp4` (maximum). Conservati anche i run intermedi.

**Stato del faceswap multi-persona**: raggiunto e verificato a occhio su due
video. Il sintomo che aveva bloccato il lavoro — «il volto torna
all'originale a intermittenza» — aveva due cause: il detector a 0.50 che
perdeva i volti ai margini e in moto, e i frame senza riga nella mappa. Un
terzo difetto, collegato, faceva assegnare la stessa foto a **tutti** i volti
di un frame senza righe, scambiando la seconda persona per errore. Lo
sfarfallio puro veniva invece da GFPGAN. Le cause si misurano con
`misura_flicker.py` e `diagnosi_difetti.py`.

Cosa resta fuori portata: `simswap` e `hififace` non sono utilizzabili con
l'ArcFace di Deep-Live-Cam. Con i soli modelli ONNX disponibili, inswapper
resta l'unica base possibile per il volto, e su un volto di 11 px non c'è
grana da ricostruire.
