# AI Face Swap — video head-swap locale

Sostituisce volti (e capelli) in un video a partire da **N foto**, in locale
su macOS/Linux oppure su **Colab/Kaggle**, senza API a pagamento.

Stack: **Deep-Live-Cam** (ex Roop) + **InsightFace** + modelli **ONNX**.
Nessun modello generativo di testo: vedi [Prompt](#prompt).

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

| Modalità | Identità residua | Zona volto | Fuori volto | Durata | Audio |
|---|---|---|---|---|---|
| originale | 0.961 | — | — | 17.16 s | sì |
| `faceswap` | **0.373** | 5.05 | 1.72 | 17.04 s | AAC 2ch |
| `headswap` | **0.214** | 30.21 | 1.80 | 17.04 s | AAC 2ch |

*Identità residua* = `1 - coseno(embedding(volto in output), embedding(foto))`.
Valore alto (~0.9) = il volto è ancora quello originale, nessuno swap.
Valore basso (~0.2) = il volto è stato sostituito.

`headswap` vince perché trasferisce anche capelli e pelle della fonte, quindi
l'identità risultante è più coerente di un volto innestato nel capello altrui.
Entrambe lasciano lo sfondo intatto (≤ 1.80/255) e preservano durata e audio.

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
| `--limit-frames N` | prova rapida sui primi N frame |
| `--save-masks DIR` | salva maschere e viste di debug |

## Come viene scelto chi è chi

**Non è possibile sapere automaticamente a chi appartiene il soggetto delle
tue foto.** L'abbinamento automatico è per frequenza di comparse nel video.

Il confronto fra una foto e un volto del video dà distanze embeddings molto
alte (0.90+): non è un errore, significa che sono persone diverse. Quindi:

1. `video_face_roster.py` raggruppa i volti del video e produce un provino
   (`persona_N_provino.png`) per ogni persona.
2. Tu riconosci le persone e o editi `swap_config.json`, oppure passi `--map`.

Su un video a due persone il risultato è 4 identità, con le due dominanti a 43
e 24 apparizioni: le altre sono frammenti dello stesso soggetto in pose diverse.

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

1. `inswapper` sostituisce il volto interno, prendendo l'identità dalla foto.
2. BiSeNet segmenta foto e video in 19 classi, separando capelli, orecchie e
   cappello da collo e abbigliamento.
3. La testa della foto viene warpiata sulla geometria del target tramite
   similar-transform dei 5 landmark.
4. Il bordo viene sfumato; **il collo è protetto** e lo sfondo non viene
   toccato.

Il colore della pelle viene allineato all'illuminazione del target, ma **solo
sui pixel della testa di riferimento**: applicarlo all'intero frame sporcherebbe
il video (in una versione intermedia sporcava il fondo di 53/255).

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