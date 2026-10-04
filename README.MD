# AI Face Swap — video head-swap locale

Sostituisce volti (e capelli) in un video a partire da **N foto**, in locale
su macOS/Linux oppure su **Colab/Kaggle**, senza API a pagamento.

Stack: **Deep-Live-Cam** (ex Roop) + **InsightFace** + modelli **ONNX**.
Nessun modello generativo di testo: vedi [Prompt](#prompt).

Documenti: [GOAL.MD](GOAL.MD) (obiettivo, vincoli, cosa non è risolto) ·
[ARCHITECTURE.MD](ARCHITECTURE.MD) (layer, modelli, algoritmo, flusso dati)

> **Repository privato.** I media di prova raffigurano persone identificabili e
> non vengono versionati (vedi `.gitignore`). `inswapper_128` è rilasciato per
> ricerca non commerciale: verifica la licenza prima di qualsiasi uso.

---

## Modalità

| Modalità | Cosa sostituisce | Modelli |
|---|---|---|
| `headswap` **(predefinita)** | volto, capelli, orecchie, cappello | BiSeNet + inswapper |
| `faceswap` | solo il volto interno | inswapper |

`inswapper` da solo lavora **dentro l'ellisse dei 5 landmark**: i capelli e le
orecchie restano quelli del video. Per un head swap serve segmentare la testa,
e questo fa BiSeNet (19 classi).

## Risultati misurati

Video di prova 720x1280, 24 fps, 409 frame, 17.04 s, macOS Intel (CoreML).

| Modalità | Identità residua | Area testa | Movimento | Fuori volto | Durata | Audio |
|---|---|---|---|---|---|---|
| originale | 0.988 | — | — | — | 17.16 s | sì |
| `headswap` (3 foto) | **0.360** | **0.98x** | **98%** | 1.36 | 17.04 s | AAC 2ch |

*Identità residua* = `1 - coseno(embedding(volto in output), embedding(foto))`.
Valore alto (~0.9) = il volto è ancora quello originale, nessuno swap.
Valore basso (~0.2) = il volto è stato sostituito.

*Area testa* = rapporto fra l'area della testa nell'output e nell'originale.
*Movimento* = quanto varia il contenuto nella testa rispetto al video
originale: 100% = si muove esattamente come il video.

L'head swap **non trapianta la testa della foto**: `inswapper` rigenera il
volto nella posa e nella posa del target, i capelli restano quelli del video e
ne viene solo allineato il colore. È il comportamento corretto: la
letteratura di riferimento (SimSwap, ACM MM 2020) osserva che introdurre i
capelli della fonte «non è desiderato, visto che stiamo sostituendo solo i
volti». Un'area testa pari a 0.98x e un movimento al 98% confermano che
niente è rimasto statico o sproporzionato.

Prima versione (rigettata in test visivo): l'area della testa oscillava da
0.36x a **15.31x** e il contenuto risultava immobile. I due difetti — maschere
BiSeNet senza vincolo di area e recolor su maschere binarie — sono corretti.

Output corrente: `~/Downloads/TestFaceSwap/headswap_3foto.mp4`.

## Migliorie qualità v3 (ottobre 2026)

### Pipeline attuale (`--quality high` di default)

1. **Maschera semantica BiSeNet** (skin + naso + occhi + bocca + labbra) al posto della sola ellisse
2. **Upscale selettivo** della regione facciale (1.6×–2.0×) → più dettaglio senza toccare tutto il frame
3. **Trasferimento colore in LAB**
4. **Poisson blending** (`seamlessClone` MIXED) con maschera semantica
5. **Temporal smoothing regionale**: occhi/naso/bocca solo al 40% del peso → meno ghosting
6. Profili pronti:

| Profilo    | face_scale | temporal | poisson | semantic | Uso tipico          |
|------------|------------|----------|---------|----------|---------------------|
| `fast`     | 1.0×       | 0.25     | no      | no       | prove rapide        |
| `high`     | 1.6×       | 0.38     | sì      | sì       | **default**         |
| `maximum`  | 2.0×       | 0.42     | sì      | sì       | massima qualità     |

```bash
# Qualità alta (default, consigliato sul tuo Mac)
python headless_faceswap.py --video clip.mp4 --photo foto.jpg --quality high

# Massima qualità (un po' più lento)
python headless_faceswap.py --video clip.mp4 --photo foto.jpg --quality maximum

# Prove veloci
python headless_faceswap.py --video clip.mp4 --photo foto.jpg --quality fast --no-enhancer
```

Su Intel UHD 630 un video di ~17 s a 24 fps richiede tipicamente:
- `high` + enhancer → **12–22 min**
- `maximum` + enhancer → **18–30 min**

Resta ampiamente sotto l'ora.

## Installazione

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git dlfolder
python3 -m venv dlfolder/venv
./dlfolder/venv/bin/pip install -r dlfolder/requirements.txt
```

Il repository è **privato** e contiene gli script, non `dlfolder/`, che resta
escluso perché contiene venv e modelli oltre 100 MB.

Al primo avvio vengono scaricati da soli:

- `inswapper_128.onnx` (554 MB), `gfpgan-1024.onnx` (se serve)
- `bisenet_resnet18.onnx` (51 MB, MIT, per l'head swap)
- `buffalo_l` di InsightFace, nella cache utente

##Uso

### Locale

```bash
PY=./dlfolder/venv/bin/python

# 1. censimento: quante persone sono nel video, con provini
$PY video_face_roster.py --video clip.mp4 --out ./censimento \
    --photo foto1.jpg foto2.jpg

# 2. sostituzione, leggendo tutto dal config
$PY headless_faceswap.py --config ./censimento/swap_config.json \
    --output ./risultato.mp4

# oppure tutto in una riga
$PY headless_faceswap.py --video clip.mp4 --photo foto1.jpg foto2.jpg \
    --mode headswap --output ./risultato.mp4
```

### Colab

`TestFaceSwap.ipynb` contiene le stesse chiamate: le celle eseguono gli script
con `subprocess`, quindi non c'è logica duplicata e il notebook non può
diventare obsoleto rispetto alla CLI.

### Opzioni utili

| Flag | Effetto |
|---|---|
| `--mode headswap\|faceswap` | predefinita `headswap` |
| `--map 0,1,0` | fissa quale foto per ogni persona (indice foto) |
| `--identities N` | forza il numero di persone distinte nel video |
| `--prompt "..."` | registra il prompt nel config (vedi sotto) |
| `--no-enhancer` | salta GFPGAN, più veloce |
| `--provider cuda\|coreml\|cpu` | forza il provider ONNX |
| `--temporal 0.35` | smorzamento temporale contro il flickering (0 = off) |
| `--harmonize 1.0` | allineamento del tono pelle (0 = off) |
| `--limit-frames N` | prova rapida sui primi N frame |
| `--save-masks DIR` | salva maschere e viste di debug |

## Come viene scelto chi è chi

**Non è possibile sapere automaticamente a chi appartiene il soggetto delle
tue foto.** Il confronto fra una foto e un volto del video dà distanze
embedding molto alte (0.90+): non è un errore, significa che sono persone
diverse.

Le persone sono numerate **in ordine di prima apparizione** nel video:/persona
1` è la prima che si vede, `persona 2` la successiva, e così via. Le foto
vengono abbinate in quello stesso ordine. Nessun soggetto è codificato nel
codice: con N foto e M persone si abbinano le prime N, e `--map` permette di
correggere a mano.

1. `video_face_roster.py` raggruppa i volti del video e produce un provino
   (`persona_N_provino.png`) per ogni persona.
2. Tu riconosci le persone dal provino e passi `--map` (indici delle foto
   in ordine di persona), oppure editi `swap_config.json`.

Il numero di persone trovate non è il numero di persone reali: su questo
video il clustering dà 5 identità per 2-3 soggetti, perché le embedding
 cambiano sensibilmente con la posa. Le identità marginali (poche
apparizioni) sono di solito frammenti dello stesso soggetto.

### Raggruppamento

Il clustering non usa una soglia fissa, che su questo video dava risultati
diversi a ogni prova:

1. le embedding vengono **medie su tracciati temporali** (i volti sono
   collegati fra frame consecutivi per IoU del bounding box);
2. la gerarchia agglomerativa viene tagliata con il **metodo di Otsu**, che
   massimizza la varianza fra le due classi e trova la valle fra "stessa
   persona" e "persone diverse".

Su questo video: 79 identità con una soglia greedy a 0.55, 26 con Otsu sulle
embedding grezze, **4** con le embedding mediate sui tracciati.

## Verifica

Non dare per scontato che sia riuscito. Un risultato può sembrare plausibile
ed essere una sovrapposizione statica, o un'immagine distrutta.

```bash
$PY verifica_swap.py --videoorig clip.mp4 --videoswap risultato.mp4 \
    --rif foto1.jpg --modalita headswap
```

Esce con codice diverso da zero se il volto non è stato sostituito, se lo
sfondo è stato toccato o se la zona del volto è distorta.

**Confronto tra video già codificati: attenzione ai numeri falsi.** Con x264 i
blocchi vicini a una zona molto cambiata vengono riquantizzati diversamente: una
differenza media di 32/255 su una fascia sottile può essere solo codice. Per
questo `verifica_swap.py` usa come metrica primaria la distanza di identità,
che non dipende dal codec.

## Prompt

I modelli ONNX usati — `inswapper`, GFPGAN, BiSeNet — **non accettano prompt di
testo**. Un prompt per head swap ("sostituisci il volto e i capelli…")
descriverebbe un setup diverso: diffusione con ControlNet/IP-Adapter.

Il prompt viene comunque registrato in `swap_config.json` con il flag
`"prompt_is_used_by_model": false`, come documentazione di riferimento.

## Come funziona l'head swap

1. `inswapper` sostituisce il volto interno, prendendo l'identità dalla foto e
   **rispettando la posa e il movimento del target**.
2. BiSeNet segmenta foto e video in 19 classi, separando capelli e pelle da
   collo e abbigliamento.
3. Il colore di capelli e pelle viene allineato a quello della foto **riscalando
   e traslando i pixel**: la struttura spaziale resta quella del video, quindi
   un capello che si muove continua a muoversi e cambia solo il colore.
4. La miscelazione passa per una maschera sfumata, non per una selezione
   binaria: con una soglia il ricolor lasciava un bordo netto, visibile come un
   contorno attorno a capelli e pelle.
5. Il bordo viene sfumato, il **collo è protetto** e lo sfondo non viene toccato.
6. `--temporal` (attivo, 0.35) mescola il fotogramma precedente, portato nella
   posizione attuale con i landmark, per ridurre il flickering.

### Dettagli tecnici che hanno fatto la differenza

- **`emap`**: l'embedding del volto di riferimento va moltiplicato per una
  matrice 512×512 salvata come ultimo initializer del grafo ONNX, poi
  rinormalizzato. Senza, il modello non riceve l'identità e l'output resta il
  volto target pressoché invariato.
- **Normalizzazione**: l'input è RGB diviso per 255, quindi in `[0, 1]`
  (equivalente a `blobFromImage` con `swapRB=True`, `mean=0`, `std=255`).
  Non `[-1, 1]`: la conversione sbagliata sballa i colori e l'identità non
  viene più riconosciuta.
- **Divisione per zero**: il ricolor calcola la deviazione standard della
  regione; se è zero (pixel uniformi) produce NaN e pixel corrotti. Va protetta
  con un epsilon.

### Modelli alternativi valutati

`swappers.py` implementa un adapter per scambiatori intercambiabili. Sono
state scaricate e provate le varianti `simswap_256`, `hififace_256` e
`gpen_bfr_256` (encoder) dagli asset FaceFusion.

`inswapper` funziona. **`simswap` e `hififace` non sono utilizzabili** con
l'identificatore di Deep-Live-Cam: non è un problema di normalizzazione
(verificate tutte le varianti) ma un mismatch di modello, perché sono stati
addestrati con un ArcFace diverso e l'embedding non è compatibile. Per usarli
servirebbe il loro modello di riconoscimento, non disponibile negli asset
consultati. `swappers.py` resta un modulo sperimentale non usato dalla
pipeline.

La stabilità temporale è il limite noto della letteratura: "Temporal
Optimization for Face Swapping Video based on Consistency Inheritance"
(ACM MM 2024) la tratta come un disturbo nel dominio del tempo e la risolve
riaddestrando il modello, che qui non è pratico. `--temporal` ne recupera una
parte senza modelli aggiuntivi.

### Limite della verifica

`verifica_swap.py` accetta **un solo** `--rif`. Con più foto va eseguito una
volta per persona, altrimenti dà esiti fuorvianti: confronta un volto
sostituito con la foto di un'altra persona e segnala un falso fallimento. Non
è ancora legato al `--map` usato per generare il video.

## Limiti noti

- Le embedding di uno stesso volto variano fino a ~0.56 fra frame consecutivi
  (posa, espressione, motion blur, compressione): da qui la necessità di
  mediare sui tracciati.
- I frame vengono estratti come PNG su disco, quindi un video molto lungo
  occupa spazio. Conviene lavorarci in segmenti.
- Se nel video un volto è troppo piccolo, di profilo o coperto, non viene
  sostituito. Lo script esce con codice 3 se non ha sostituito nessun volto,
  invece di produrre un video identico all'originale.

## Note di audit

`final_result_3subjects.mp4` non è un face swap: è il video sorgente con 3 foto
statiche incollate al centro e cambiate ogni 5 s. La zona centrale è immobile
(MAD 0.153/255) mentre il bordo si muove (4.679/255), e l'output era troncato a
360 frame su 409. Il face swap reale è quello prodotto da `headless_faceswap.py`.