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

## Uso

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
| `--mappa-file CSV` | **consigliato**: foto per frame, da `abbina_volti.py` |
| `--map 0,1,0` | fissa quale foto per ogni persona (indice foto) |
| `--det-size 1024` | ingresso del detector: più alto = volti piccoli trovati meglio |
| `--identities N` | forza il numero di persone distinte nel video |
| `--no-enhancer` | salta GFPGAN: **necessario** in faceswap, vedi sotto |
| `--swapper inswapper\|hyperswap\|hififace` | modello di sostituzione |
| `--stabilizza 0.6` | smorza i keypoint del detector (0 = off) |
| `--temporal 0.35` | smorzamento temporale (0 = off) |
| `--crf 17` | qualità x264 (più basso = migliore) |
| `--zoom 1.4` | ingrandisce il ritaglio prima dello swap |
| `--prompt "..."` | registra il prompt nel config (vedi sotto) |
| `--provider cuda\|coreml\|cpu` | forza il provider ONNX |
| `--harmonize 1.0` | allineamento del tono pelle (0 = off) |
| `--limit-frames N` | prova rapida sui primi N frame |
| `--save-masks DIR` | salva maschere e viste di debug |
| `--keep-frames` | non cancella i PNG estratti |

## Faceswap multi-persona: `abbina_volti.py` → mappa

Con più persone il numero trovato dal clustering **non è affidabile**: due
algoritmi diversi possono contare identità diverse, e `--map` finirebbe sugli
indici sbagliati con un risultato plausibile ma sulla persona errata. La strada
sicura è passare la mappa per **frame**.

```bash
PY=./dlfolder/venv/bin/python

# 1. censimento + assegnazione (indici 0-based nell'ordine di --photo)
$PY abbina_volti.py --video clip.mp4 \
    --photo foto3.jpeg foto1.jpeg foto2.jpeg \
    --out ./match_clip --det-size 1024 \
    --min-rilevamenti 1 --min-px 0 \
    --assegna 1,1,2,0,0,0,0,1 --provider cpu

# 2. swap leggendo la mappa per frame (clustering interno ignorato)
$PY headless_faceswap.py --mode faceswap \
    --video clip.mp4 --photo foto3.jpeg foto1.jpeg foto2.jpeg \
    --mappa-file ./match_clip/match_frame_foto.csv \
    --det-size 1024 --no-enhancer --output ./out.mp4
```

`abbina_volti.py` produce:

| file | ruolo |
|---|---|
| `persona_N.png` | griglia di volti campione: riconosci chi è chi |
| `match_riepilogo.md` | tabella leggibile con somiglianze e intervalli frame |
| `match_frame_foto.csv` | `frame,persona,x1,y1,x2,y2,sim_*,foto_assegnata,foto_indice` |
| `match_config.json` | config pronto per `headless_faceswap.py` |

- **`foto_indice` è 0-based sull'ordine di `--photo`**; `-1` lascia il volto
  originale. Attenzione: `0` è un indice valido, non un "mancante".
- **`--det-size` deve essere uguale** nei due script: i bbox del CSV sono
  prodotti a quella dimensione e l'abbinamento per bbox non combacia se le due
  corrono diverse.
- **`--min-rilevamenti` / `--min-px`** decidono quali cluster contano come
  persone visibili. `1` e `0` li disattivano: serve quando un cluster reale
  ha poche apparizioni, altrimenti i frame di quella persona restano
  scambiati per originale.
- Le somiglianze nel CSV sono riportate, mai usate per decidere: se il
  soggetto della foto non è nel video sono rumore (0.0–0.2).

**Come un volto sceglie la propria foto.** Per ogni volto rilevato nel frame:

1. si prende la riga del CSV con **IoU maggiore**, ma solo se `≥ 0,30`;
2. se il frame non ha righe, si guardano i **6 frame adiacenti** e si sceglie
   la riga più vicina, con distanza normalizzata sulla **larghezza del volto
   rilevato** — una larghezza è il limite. Sopra, il volto è un'altra
   persona: la seconda persona di un piano resta intatta invece di prendere
   la foto del soggetto principale;
3. se nessuna riga supera la soglia, il volto resta com'è e viene contato nel
   log (`Altre: N volti lasciati com'e'`).

Il limite è che il detector deve **trovare** il volto: sui margini e in moto
perdeva soggetti con punteggio 0.26–0.47 sotto la soglia di 0.50. In
`--mode faceswap` la soglia viene abbassata a 0.25 per tutti i frame
(`_soglia_ridotta` in `headless_faceswap.py`): la mappa decide comunque chi
scambiare, quindi non cambia l'identità di ciò che viene sostituito. In
`headswap` resta 0.50, dove la soglia partecipa al tracciamento. La modifica
sta nel nostro script e non in `dlfolder/`, che è una repo separata esclusa
dal versionamento.

## Come viene scelto chi è chi

**Non è possibile sapere automaticamente a chi appartiene il soggetto delle
tue foto.** Il confronto fra una foto e un volto del video dà distanze
embedding molto alte (0.90+): non è un errore, significa che sono persone
diverse.

Le persone sono numerate **in ordine di prima apparizione** nel video: `persona
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
---

## Verifica oggettiva: tre strumenti nuovi

Fin qui i difetti si guardavano a occhio, il che non scala e non dà
numeri. Esistono ora tre misure automatiche.

### `diagnosi_difetti.py` — melting, warping, allineamento

```bash
./dlfolder/venv/bin/python diagnosi_difetti.py --originale VIDEO --output OUT.mp4
```

| difetto | firma misurata |
|---|---|
| face melting | varianza del Laplaciano nella regione del volto rispetto all'originale |
| warped face | distanza inter-oculare e rapporto bocca/occhi sui landmark |
| alignment failure | spostamento del baricentro dei landmark |
| bordo visibile | eccesso di gradiente sul contorno rispetto all'originale |
| flickering | deviazione standard dell'identità fra frame |

Tre correzioni rese necessarie dai dati, non da supposizioni:

- **Il bordo non era un difetto.** La prima versione confrontava il
  gradiente con una soglia assoluta, ma il contorno del viso ha già
  gradiente naturale (zigomi, mascella): misurava +30 e sembrava un difetto
  grave. Misurando l'*eccesso* rispetto all'originale il valore reale è
  +0.4, cioè nulla. Sarebbe stato un fix su un problema inesistente.
- **Il warping era rumore.** Su un volto di 11 px la regressione dei landmark
  dà scarti del 100-160%. La geometria ora si misura solo dove la distanza
  inter-oculare supera i 25 px.
- **Lo spostamento andrà misurato sui landmark, non sul bounding box.** In un
  head swap il detector dimensiona il box anche sui capelli, quindi il suo
  centro si sposta di 7-8 px *anche quando il volto è allineato*. Il
  baricentro dei landmark si sposta di 2.3 px.

### `misura_flicker.py` — sfarfallio frame per frame

```bash
./dlfolder/venv/bin/python misura_flicker.py --video ORIGINALE OUT.mp4 \
    --da 0 --a 120
```

Confronta l'identità del volto fra un frame e il successivo, nel video
scambiato e in quello originale. **Si legge solo l'eccesso**: l'originale ha
di suo una variazione di ~0.04–0.05, quindi 0.05 nello swap non significa
nulla. Soglie: spostamento 6 px, eccesso di bordo +6, overlap 85%.

### Il flickering aveva due cause, entrambe trovate coi numeri

**1. GFPGAN.** L'enhancer non è neutro: spingeva i dettagli della pelle al
400% dell'originale e quel contrasto in più, sul video non swappato, si legge
come sfarfallio. È il motivo per cui tutte le run di faceswap usano
`--no-enhancer`, non solo per velocità.

**2. Il detector che perde e ritrova il volto.** Con la soglia a 0.50 i volti
ai margini e in moto uscivano con punteggio 0.26–0.47 e venivano scartati: il
frame tornava all'originale, quello dopo riprendeva lo swap. Effetto
lampeggiante identico a un difetto di blending, ma era un problema di
rilevamento. Corretto abbassando la soglia in faceswap (vedi
`_soglia_ridotta`).

Una terza falsa pista: `--stabilizza` (default 0) e `--temporal` smorzano i
movimenti, ma non correggono un volto che sparisce. Applicarli su questo
sintomo maschera la causa.

### `verifica_headswap.py` — distingue head swap da face swap

```bash
./dlfolder/venv/bin/python verifica_headswap.py --originale VIDEO --output OUT.mp4
```

Misura quale percentuale di pixel è stata riscritta, regione per regione,
usando le classi di BiSeNet. È la verifica che risponde alla domanda se il
risultato è un head swap o un face swap: se i capelli non cambiano, è un face
swap, per quanto bene rifinito sia il volto.

## I capelli: perché il risultato era un face swap

Il ricoloramento dei capelli non può produrre un head swap: i capelli
restano quelli del video e si legge come un face swap. Ora la capigliatura
della foto viene allineata al volto del target e incollata, limitata a una
regione geometrica e tagliata sotto il mento, con i colori riallineati
all'illuminazione del target.

Un dettaglio che è costato una diagnosi: la maschera dei capelli della fonte
**non** va intersecata con quella del target. La maschera del target descrive
la capigliatura già presente nel video e non può mai accogliere una
pettinatura diversa: intersecandole si perdeva il 92% dei capelli.

## Risultati

Video 720x1280, 24 fps, 409 frame, tre foto, `--map 0,1,2,0,0`.

| | v3.1 high | test1 fast | test2 high | test3 maximum |
|---|---|---|---|---|
| dettagli del volto | 90% | 104% | 116% | 151% |
| distanza inter-oculare | 3.69% | 3.69% | 3.64% | 3.88% |
| proporzioni bocca/occhi | 2.60% | 3.31% | 2.59% | 3.07% |
| spostamento landmark | 2.17 px | 2.21 px | 2.30 px | 2.81 px |
| flickering | 0.120 | 0.039 | 0.054 | 0.058 |
| **capelli riscritti** | 6.4% | **32.4%** | **33.4%** | **33.5%** |
| esterno toccato | 0.1% | 0.7% | 0.7% | 0.7% |
| tempo | 10.8 min | 9.1 min | 9.2 min | 9.4 min |

Nessun difetto oltre soglia sui tre test. Il melting della v3 è risolto: i
dettagli del volto sono ora *sopra* il 100%, non al 69% della v2. Il
flicker è sceso da 0.120 a 0.039-0.058. Movimento, collo, sfondo e audio
restano intatti (0.7% di pixel esterni toccati).

`test2` (high) e `test3` (maximum) sono i migliori: `test3` ha il doppio
 dei dettagli dell'originale, `test2` è il più fedele alla geometria.

## Limiti noti

- Il volto è un trapianto: identità e posa vengono dalla foto, ma la
  micropittura della pelle resta quella di inswapper. Su un volto di 11 px
  non c'è nulla da correggere e nulla da trapiantare.
- `inswapper_128.onnx` è research / non-commerciale.
- `simswap` e `hififace` non funzionano con l'ArcFace di Deep-Live-Cam:
  richiedono recognizer dedicati. L'unico modello compatibile è inswapper.
- La maschera semantica ora interseca l'ellisse invece di allargarla: allargare
  la maschera oltre i limiti del volto è la causa del melting.
- Le proporzioni si misurano solo sui frame in cui il volto supera i 25 px di
  distanza inter-oculare: sotto quella soglia i landmark non sono affidabili.
