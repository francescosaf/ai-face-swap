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

- **L'abbinamento foto↔persona è manuale.** Le embedding di una foto e di un
  volto del video distano 0.90+: sono persone diverse, non un errore. Quindi
  non si può dedurre a chi appartiene il soggetto di una foto. Serve guardare i
  provini e usare `--map`.
- **Il clustering non è garantito.** Le embedding dello stesso volto variano
  fino a ~0.56 fra frame consecutivi. Sul video di prova dà 4 identità
  (le due principali a 43 e 24 apparizioni); su un video con molta gente o
  pose molto diverse può frammentarsi.
- **I frame sono PNG su disco**: un video molto lungo occupa molto spazio.
  Conviene lavorare in segmenti.
- **Il percorso di enhancement GFPGAN non è mai stato testato**: tutte le run
  hanno usato `--no-enhancer`, quindi `gfpgan-1024.onnx` non è mai stato
  scaricato. La flag esiste e il codice è predisposto, ma non è verificata.

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

Cosa resta fuori portata: `simswap` e `hififace` non sono utilizzabili con
l'ArcFace di Deep-Live-Cam. Con i soli modelli ONNX disponibili, inswapper
resta l'unica base possibile per il volto, e su un volto di 11 px non c'è
grana da ricostruire.
