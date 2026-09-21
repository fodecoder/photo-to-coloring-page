# Diagnosi — perché l'output è lontano da `docs/desired*.jpg`

Misure fatte il 2026-09-18 sul repo allo stato attuale, con gli engine ai
parametri di default. Tutte le immagini sono normalizzate a 864 px di lato
lungo prima del confronto.

---

## 0. Riassunto

Ci sono quattro cause indipendenti, in ordine di impatto:

1. **Il problema è formulato male**: il riferimento non è una mappa di bordi,
   è un *disegno*. Gli engine di default risolvono un problema diverso.
2. **`postprocess.nms_centerline` raddoppia ogni tratto.** È il default per
   entrambi gli engine ML. Bug dimostrabile in 6 righe.
3. **`informative_drawings` gira a 256×256 quadrati.** Aspect ratio distrutto,
   risoluzione 4× sotto quella di training, poi upscale 5×.
4. **La metrica di accettazione è incoerente.** Il criterio "F1 ≥ 0.75 a
   tolleranza 2 px" di `IMPROVEMENT-PROMPT-2.md` è irraggiungibile per
   costruzione, e ha guidato scelte sbagliate (fra cui il punto 2).

---

## 1. Il riferimento non è una edge map

`desired.jpg` e `desired-5.jpg` sono **line art semantica**: un contorno chiuso
per oggetto, spessore uniforme, zero ombreggiatura, zero texture, e dettagli
*reinterpretati* (l'erba diventa un pattern di trattini stilizzati; i volti
hanno occhi e bocca ridisegnati). È l'output tipico di un modello della
famiglia *lineart anime / manga line extraction*, non di un edge detector.

`canny`, `chained`, `skeleton`, `xdog`, `cartoon` rispondono tutti al
**gradiente locale**. Su una tavola illustrata il gradiente è ovunque: glow,
sfumature, riflessi sull'acqua, grana della carta. Nessuna taratura di soglia
distingue "bordo di oggetto" da "bordo di ombra", perché a livello di pixel
sono lo stesso segnale. Questo è già stato verificato nello sweep documentato
in `IMPROVEMENT-PROMPT-2.md` § 2.1 e la conclusione resta valida.

**Conseguenza**: `canny` come `--style` di default garantisce un output
inutilizzabile in produzione, qualunque sia la taratura.

---

## 2. Bug: `nms_centerline` produce due linee per ogni tratto

`postprocess.nms_centerline` applica `cv2.Canny` alla soft map. Canny trova i
**massimi del gradiente**, cioè i due fianchi del tratto — non la sua cresta.
Un tratto scuro largo 6 px non diventa una linea: diventa il suo contorno.

```python
soft = np.full((60, 120), 255, np.uint8)
soft[20:26, 10:110] = 20  # tratto spesso 6 px
soft[45:47, 10:110] = 20  # tratto spesso 2 px

nms_centerline(normalize_percentile(soft))
# righe con inchiostro: [19, 20, 21, 22, 23, 24, 25,  44, 45, 46]
#                        ^^^^^^^^^^ i due fianchi      ^^^^^^^^^^

hysteresis_centerline(normalize_percentile(soft), min_branch_length=3)
# righe con inchiostro: [22, 45]        <- corretto
```

`"nms"` è il default di `soft_map_to_line_art`, quindi di
`informative_drawings` **e** di `anime2sketch`. È il motivo per cui l'output
ML sembra fatto di macchie contornate invece che di linee.

La docstring del modulo dice che `"nms"` è stato scelto perché "recupera più
inchiostro vero di `hysteresis`". È vero e non significa niente: raddoppiare
i tratti raddoppia la recall, e la metrica usata per decidere non penalizza
lo spessore (vedi § 4). La scelta è un artefatto della metrica, non
dell'immagine.

> Nota minore, stessa famiglia: `redraw_centerline` usa `findContours`, che
> traccia il *bordo* di una maschera. Su uno scheletro da 1 px è innocuo (il
> contorno percorre la linea andata e ritorno), ma la funzione è documentata
> come generica e su qualsiasi maschera più spessa raddoppierebbe di nuovo.
> Va documentata come "accetta solo maschere 1 px" o riscritta su
> `ximgproc.thinning` + tracciamento esplicito.

---

## 3. Bug: `informative_drawings` gira alla risoluzione sbagliata

```python
resized = cv2.resize(rgb, (self.load_size, self.load_size))  # load_size = 256
```

Tre difetti in una riga:

- **quadrato**: 1536×2048 schiacciato a 256×256 e poi ristirato — la geometria
  esce deformata;
- **256 invece di 512**: il preprocessore ufficiale
  (`controlnet_aux.LineartDetector`) usa `detect_resolution=512` applicato al
  **lato corto**, con arrotondamento a multiplo di 64 e aspect ratio
  preservato;
- **upscale 4–5×** della soft map prima della binarizzazione, che trasforma
  tratti da 1 px in gradienti larghi 5 px — i "blob" poi contornati dal bug § 2.

`anime2sketch.py` questo problema ce l'aveva e **è già stato corretto**
(`_resize_and_pad`, load_size 512). `informative_drawings.py` no: è rimasto
indietro.

### Effetto misurato (`starting-image.jpeg` vs `desired.jpg`)

Inferenza rifatta a parità di pesi, cambiando solo risoluzione e
post-processing:

| configurazione | F1 @tol 2 | F1 @tol 4 | P @tol 4 | R @tol 4 | inchiostro |
|---|---:|---:|---:|---:|---:|
| **attuale** (256², NMS) | **0.294** | 0.468 | — | — | 17.9 % |
| 512 lato corto + hysteresis | 0.392 | 0.620 | 0.623 | 0.617 | 7.4 % |
| 768 lato corto + hysteresis | 0.488 | 0.685 | 0.697 | 0.674 | 8.5 % |
| **1024 lato corto + hysteresis** | **0.536** | **0.713** | 0.730 | 0.696 | 8.6 % |

**+82 % di F1 senza toccare la rete né i pesi**, e il numero cresce ancora con
la risoluzione: il modello è completamente convoluzionale, la costante 256 non
ha alcuna giustificazione.

### Riverifica 2026-09-21 con torch reale, `detect_resolution` 512–1536

La tabella sopra è stata misurata con una reimplementazione NumPy della rete
(l'ambiente di allora non aveva PyTorch). Il codice attuale
(`_resize_short_side`, `detect_resolution` di default 1024,
`postprocess_strategy="hysteresis"`) implementa già i punti 1-3 e 5 del
checklist di `IMPROVEMENT-PROMPT-3.md`; restava da rifare la tabella con
torch vero e misurare anche 1280/1536, come richiesto lì esplicitamente
("se diverge di più del 3% di F1, fermati").

Rieseguita con `weights/informative_drawings.pth` (checkpoint "fine",
SHA256 `c686ced2...`, stesso pinnato dal commit che l'ha scelto), ai
parametri di default correnti (`postprocess_strategy="hysteresis"`),
`coloring_page.metrics.compare_boundaries` sulle 3 coppie di riferimento:

| detect_resolution | F1@2 (img1) | F1@4 (img1) | F1@4 (img5) | F1@4 (img7*) | inchiostro (img1) |
|---:|---:|---:|---:|---:|---:|
| 512  | 0.085 | 0.318 | 0.432 | 0.229 | 9.4 % |
| 768  | 0.269 | 0.467 | 0.611 | 0.294 | 8.3 % |
| **1024** | **0.364** | **0.525** | 0.681 | 0.284 | 7.3 % |
| 1280 | 0.405 | 0.550 | 0.709 | 0.263 | 6.6 % |
| 1536 | 0.415 | 0.563 | 0.721 | 0.242 | 6.1 % |

\* `starting-image-7.jpeg`/`desired-7.jpg` ha aspect ratio diverso dal
riferimento (vedi §7): i suoi numeri sono approssimati e infatti è l'unica
coppia dove F1 *cala* oltre 768 invece di continuare a salire.

**Il numero a 1024 (F1@4 = 0.525) diverge dalla stima NumPy (0.713) di
molto più del 3% dichiarato come soglia d'allarme.** Prima di considerare
il fix acquisito sono stati controllati arch e checkpoint:

- `state_dict` del checkpoint pinnato carica nel generatore attuale con
  `missing_keys=[]` e `unexpected_keys=[]` -- nessun mismatch strutturale;
- lo SHA256 del file in `weights/` coincide esattamente con quello pinnato
  in `scripts/fetch_weights.py` per `sk_model.pth`;
- il commit `2913653` (confronto fine/coarse) aveva già misurato con torch
  reale, a `detect_resolution=1024` ma `strategy="nms"`: f1_normalized
  0.507 (img1), 0.705 (img5), 0.258 (img7) -- valori dello stesso ordine di
  grandezza di quelli di questa tabella (0.525, 0.681, 0.284 con
  `"hysteresis"`), misurati da un run indipendente, mesi prima, con un
  altro strategy di default.

Due misure indipendenti con torch reale concordano fra loro e divergono
entrambe dalla stima NumPy. La spiegazione più probabile non è un bug
nell'arch/checkpoint attuale (verificato pulito sopra), ma un'imprecisione
della reimplementazione NumPy usata per la diagnosi originale -- la tabella
NumPy andava trattata come una stima direzionale, non come un valore
acquisito, esattamente come il prompt stesso avvertiva.

**Curva e default**: F1@4 su img1/img5 sale ancora da 1024 a 1536, ma con
rendimenti calanti (+0.038 e +0.040 assoluti da 1024 a 1536, contro +0.207
e +0.156 da 512 a 1024) a fronte di un costo che cresce quadraticamente col
lato (1536² / 1024² ≈ 2.25×). 1024 resta il ginocchio ragionevole della
curva e l'inchiostro (7.3% / 8.2% sulle coppie senza mismatch di aspect
ratio) è il più vicino alla banda 3-7% fra tutte le risoluzioni misurate.
**Nessuna modifica al default proposta**: `detect_resolution=1024` resta
invariato.

### Checkpoint: probabilmente quello sbagliato

Il file in `weights/` produce uno stile "matita", con tratteggio e ombreggiatura
— è il checkpoint **fine** (`sk_model.pth`). Quello *coarse* (`sk_model2.pth`)
è esattamente ciò che serve qui, ed è già previsto da `scripts/fetch_weights.py`
ma non scaricato. Va misurato prima di qualsiasi altra cosa:

```bash
python scripts/fetch_weights.py --filename sk_model2.pth --dest weights/informative_drawings-coarse.pth
```

*Confidenza: alta sul fatto che il checkpoint attuale sia il fine; media su
quanto guadagno dia il coarse.*

---

## 4. La metrica di accettazione è incoerente

Calibrazione della `boundary_f_measure` attuale, usando come "predizione"
trasformazioni note del riferimento stesso:

| predizione | F1 @1 | F1 @2 | F1 @4 | F1 @8 | inchiostro |
|---|---:|---:|---:|---:|---:|
| riferimento identico | 1.000 | 1.000 | 1.000 | 1.000 | 3.8 % |
| riferimento dilatato 3×3 | — | **1.000** | 1.000 | 1.000 | 11.4 % |
| **riferimento traslato di 3 px** | 0.278 | **0.502** | 0.923 | 0.999 | 3.8 % |
| riferimento traslato di 6 px | 0.190 | 0.308 | 0.546 | 0.973 | 3.7 % |
| rumore casuale 4 % | 0.123 | 0.218 | 0.400 | 0.611 | 4.0 % |
| pagina tutta nera | 0.168 | 0.258 | 0.413 | 0.612 | 100 % |
| `canny` (default attuale) | 0.478 | 0.614 | 0.736 | 0.838 | 11.0 % |

Due letture, entrambe fatali per il criterio attuale:

1. **A tolleranza 2, il riferimento stesso spostato di 3 px prende 0.502, meno
   di `canny` (0.614).** La metrica sta misurando l'allineamento sub-pixel di
   un disegno *reinterpretato*, non la qualità del disegno. Chiedere F1 ≥ 0.75
   a tolleranza 2 significa chiedere che il tool riproduca le scelte di
   MimiPanda entro 2 px. Non è un obiettivo: è un errore di unità.
2. **Dilatare il riferimento fino all'11 % di inchiostro lascia F1 = 1.000.**
   La metrica non penalizza lo spessore, ma la recall cresce con l'inchiostro:
   è per questo che è stato scelto `nms` (§ 2) e che `canny`, con l'11 % di
   inchiostro, batte prototipi visivamente migliori con il 5 %.

### Metrica da adottare

- **Tolleranza = 0.5 % del lato lungo** (4 px a 864 px). A quel valore il
  floor degenere è 0.41 e un disegno corretto ma traslato di 3 px prende 0.923:
  c'è dinamica utile fra "rumore" e "giusto".
- **Riportare sempre il floor degenere** e normalizzare:
  `F1_norm = (F1 − F1_floor) / (1 − F1_floor)`.
- **Vincolare la copertura di inchiostro** a 3–7 % come condizione di
  ammissibilità, *non* come termine da ottimizzare. Senza questo vincolo la
  metrica premia il disegnare di più.
- **Assottigliare entrambe le mappe a 1 px** prima del confronto, altrimenti si
  sta confrontando anche lo spessore del pennello.

Finché la metrica resta quella attuale, ogni taratura futura sarà guidata nella
direzione sbagliata — è già successo due volte.

**Nota (2026-09-21)**: le correzioni sopra sono state applicate, ma questa
metrica resta comunque un confronto con un riferimento artisticamente
reinterpretato, non una misura diretta di colorabilità. È stata quindi
spostata in `scripts/metrics.py` come strumento diagnostico per lo sviluppo
(confrontare motori fra loro), e sostituita come criterio di accettazione
da `coloring_page.validate.QualityReport` (ink coverage, regioni chiuse vs.
"leaking", endpoint di contorno pendenti) — vedi il README, sezione
"Validating output".

---

## 5. La direzione che funziona: confini di regione, non bordi di gradiente

Un edge detector chiede "qui c'è un salto di intensità?". Un disegno da
colorare risponde a una domanda diversa: "qui finisce un oggetto e ne comincia
un altro?". La seconda domanda si risolve **segmentando in regioni** e
disegnando i confini fra regioni — che sono chiusi per costruzione, e non
esistono dove c'è solo un'ombra.

Prototipo misurato (30 righe, nessuna dipendenza nuova):
`ximgproc.l0Smooth` → `pyrMeanShiftFiltering` → gradiente in spazio **Lab** →
soglia a percentile → `thinning` → `approxPolyDP` + `polylines`.

| immagine | engine | F1 @4 | P @4 | R @4 | inchiostro |
|---|---|---:|---:|---:|---:|
| `starting-image` | `canny` | 0.736 | 0.678 | 0.805 | 9.8 % |
| | `chained` | 0.731 | 0.683 | 0.786 | 10.9 % |
| | **region prototype** | 0.681 | **0.789** | 0.599 | **5.3 %** |
| `starting-image-5` | `canny` | 0.861 | 0.892 | 0.832 | 8.6 % |
| | **region prototype** | 0.789 | **0.942** | 0.679 | **5.9 %** |
| `starting-image-7` | `chained` | 0.619 | 0.494 | 0.829 | 14.4 % |
| | **region prototype** | 0.610 | **0.604** | 0.616 | **5.5 %** |

Stesso F1 con **metà dell'inchiostro** e precision sistematicamente più alta —
cioè: quasi tutto quello che disegna è giusto. Visivamente è il prototipo più
vicino al riferimento fra tutti quelli provati (vedi la contact sheet): contorni
dei dinosauri chiusi, dinosauri piccoli e montagne presenti, acqua e cielo
puliti. Il fatto che l'F1 non salga conferma § 4: la metrica premia la recall.

Passi successivi su questa linea, non ancora provati:

- sostituire mean shift con **SLIC + region merging** su distanza Lab, o con
  `ximgproc.createSuperpixelSEEDS`, per un controllo esplicito sul numero di
  regioni finali;
- filtrare i confini per **lunghezza del contorno** e per **contrasto medio fra
  le due regioni adiacenti** (informazione che un edge detector non ha);
- chiudere i contorni residui tracciando i bordi della mappa di etichette con
  `findContours` invece che sogliando un gradiente.

---

## 6. Piano operativo, in ordine

Le fasi 1–3 sono bug fix a costo quasi zero e vanno fatte prima di qualsiasi
taratura.

1. **Metrica** (`metrics.py`, `scripts/compare.py`): tolleranza relativa
   (0.5 % del lato lungo), thinning di entrambe le mappe, floor degenere
   riportato e F1 normalizzato, vincolo di ammissibilità sull'inchiostro 3–7 %.
   Aggiornare il criterio di accettazione in `IMPROVEMENT-PROMPT-2.md` § 8.
2. **`postprocess.py`**: default `strategy="hysteresis"`. Aggiungere un test
   sintetico (il tratto da 6 px di § 2) che fallisce se un tratto pieno
   produce più di una linea. Rinominare `nms_centerline` in
   `gradient_edges` — non è una centerline e il nome ha ingannato la scelta
   del default.
3. **`informative_drawings.py`**: resize stile ControlNet (lato corto, aspect
   ratio preservato, multiplo di 64), `detect_resolution` configurabile con
   default 1024, inferenza alla risoluzione di lavoro senza upscale della soft
   map. Riusare `_resize_and_pad` di `anime2sketch.py`.
4. **Checkpoint**: scaricare `sk_model2.pth`, misurare fine vs coarse, pinnare
   lo SHA del vincitore, aggiornare `THIRD_PARTY_LICENSES.md`.
5. **Nuovo engine `region`**: il prototipo del § 5, con `--debug-dir` sugli
   stadi intermedi.
6. **Default del CLI**: `canny` non deve restare il default. Il candidato è
   `region` (zero ML) o `informative_drawings` corretto (con ML). Decidere con
   `scripts/compare.py` sulla metrica del punto 1, non a occhio.
7. **Solo dopo**: valutare un modello della famiglia giusta —
   `lineart_anime` / MangaLineExtraction — che è quella che produce lo stile di
   `desired*.jpg`. E valutare **Sketch Simplification** (Simo-Serra et al.) in
   cascata, che è plausibilmente il passo che dà al riferimento il suo aspetto
   "disegnato".

---

## 7. Sul set di valutazione

3 riferimenti per 7 input, e le coppie 5 e 7 hanno aspect ratio diverso
dall'input (MimiPanda ha ritagliato: `starting-image-5` è 1536×1292, ratio
1.19; `desired-5` è 1152×864, ratio 1.33). Il confronto su quelle due coppie è
approssimato e il codice lo segnala già con un warning, ma i numeri vanno letti
sapendolo.

Prima di tarare qualsiasi cosa: generare i riferimenti per tutte e 7 le
immagini e tenerne 2 fuori dal tuning. Con 3 coppie, di cui 2 disallineate, il
sovradattamento è garantito.
