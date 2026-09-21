# Prompt per Claude Code — rifacimento della qualità di conversione

> Incolla tutto il contenuto sotto la riga in una sessione di Claude Code aperta
> nella root di `photo-to-coloring-page`.

---

## Contesto

Il repo converte foto in line art da colorare. L'output attuale è inutilizzabile.
Misure fatte su `docs/starting-image.jpeg` (1536×2048, foto di una pagina di libro
illustrato) confrontate con `docs/desired.jpg` (output di riferimento MimiPanda:
tratto nero uniforme, contorni chiusi, zero texture, zero ombreggiatura):

| engine   | copertura inchiostro | componenti connesse | giudizio |
|----------|---------------------:|--------------------:|----------|
| canny    | 2.47 %               | 510                 | contorni spezzati, tratto 1px |
| adaptive | 25.97 %              | 7515                | stipple illeggibile |
| xdog     | **0.05 %**           | 228                 | praticamente pagina bianca |
| cartoon  | 8.91 %               | 917                 | il migliore, ma rumoroso |
| *target* | *~5–8 %*             | *~100–200*          | |

Obiettivo: tratto di larghezza uniforme, contorni connessi, texture pittorica
soppressa. Lavora per fasi, in commit separati, verificando le metriche a ogni
passo. Rispetta `CLAUDE.md` (docstring NumPy in inglese, type hints, test in
`tests/` speculari a `src/`, nessun trailer di attribuzione AI nei commit).

---

## Fase 1 — Bug fix (nessuna nuova dipendenza)

### 1.1 `engines/xdog.py` — la formula XDoG è sbagliata

Tre difetti indipendenti:

1. `diff = g1 - p * (g1 - g2)` espande a `(1-p)*g1 + p*g2`. La formula di
   Winnemöller et al. 2012 è il **DoG affilato**: `(1 + p) * g1 - p * g2`.
   Il segno è invertito.
2. Manca il parametro di nitidezza `φ` nella soglia morbida. Il paper definisce
   `T(u) = 1 se u ≥ ε, altrimenti 1 + tanh(φ · (u - ε))`. Il codice usa
   `tanh(diff - ε)`: su un intervallo così stretto `tanh` è quasi lineare e non
   produce mai nero.
3. `ε = -0.1` è nel dominio sbagliato. Misurato: `diff` ha media 0.533 e il
   **99.6 % dei pixel è ≥ ε**, quindi diventa bianco. Con `σ = 0.5` e `kσ = 0.8`
   le due gaussiane sono quasi identiche, la differenza è ~0 e `diff ≈ g`.

Riscrivi con `(1+p)*g1 - p*g2`, aggiungi `phi: float = 40.0`, e porta i default a
`sigma=1.0, k=1.6, p=25.0, epsilon=0.95, phi=40.0`. **`sigma` va scalato con la
risoluzione di lavoro** (vedi 1.3), non fissato in pixel assoluti.
Aggiungi un test che verifica che la copertura di inchiostro su un'immagine
sintetica cada in una banda ragionevole (es. 1–30 %), così una regressione a
"pagina bianca" fallisce.

### 1.2 `pipeline.py::remove_small_specks` — complessità e criterio

Due problemi:

- **Performance**: `cleaned[labels == label] = 255` in un loop fa una scansione
  completa dell'immagine per ogni componente. Con 7515 componenti su 3 Mpx sono
  ~23 miliardi di confronti. Sostituisci con una LUT booleana sulle label:
  ```python
  keep = stats[:, cv2.CC_STAT_AREA] >= min_area
  keep[0] = True  # background
  cleaned = np.where(keep[labels], binary_image, 255)
  ```
- **Criterio sbagliato**: l'area è la metrica sbagliata per i tratti. Una linea
  lunga e sottile ha area piccola quanto una macchia. Filtra invece su
  `max(CC_STAT_WIDTH, CC_STAT_HEIGHT)` (estensione) oppure sulla lunghezza dello
  scheletro. Inoltre `min_area=4` su un'immagine da 3 Mpx non rimuove nulla:
  la soglia deve essere **relativa alla dimensione dell'immagine**.

Rinomina in `remove_short_strokes` mantenendo l'API vecchia come alias deprecato,
o aggiorna tutti i call site.

### 1.3 Normalizzazione della risoluzione (causa principale del rumore)

Tutti gli engine girano alla risoluzione nativa con kernel in pixel fissi
(`bilateralFilter d=9`, `block_size=9`, morfologia 3×3, `min_area=4`). A 2048 px
la grana della carta fotografata è enorme rispetto a quei kernel: è questo che
genera i 7515 frammenti di `adaptive`. `--max-dimension` esiste ma il default è
`None`.

- Porta il default di `--max-dimension` a **1200–1600**.
- Introduci in `pipeline.py` un concetto esplicito di *working resolution*: il
  processing avviene sempre a lato lungo normalizzato, e l'output viene
  riportato alla dimensione richiesta alla fine. Documenta che gli engine
  possono assumere quella scala.
- Deriva i parametri di kernel dalla dimensione di lavoro invece di hardcodarli.

### 1.4 `engines/canny.py` — soglie auto sbagliate per questo dominio

L'euristica "auto-Canny" sulla mediana dell'intensità è pensata per foto a
istogramma centrato. Su una pagina chiara (mediana ~200) produce
`low=134, high=255`: quasi nessun bordo tenue passa, da cui i contorni spezzati.

Sostituisci con soglie derivate dalla **distribuzione della magnitudine del
gradiente**, non dall'intensità:

```python
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
magnitude = cv2.magnitude(gx, gy)
high = float(np.percentile(magnitude, 97.5))
low = high * 0.4
edges = cv2.Canny(gray, low, high, L2gradient=True)
```

Nota `L2gradient=True`: attualmente manca e la norma L1 sovrastima i gradienti
diagonali.

### 1.5 `--thickness` default 1

Un tratto da 1 px su una pagina A4 stampata è un capello: illeggibile e
impossibile da colorare dentro. Il riferimento ha ~2–3 px a 864 px di larghezza.
Porta il default a 2 e rendi lo spessore **proporzionale alla dimensione di
lavoro** (es. `max(2, round(working_size / 500))`), non assoluto.

### 1.6 `engines/anime2sketch.py` — resize quadrato distrugge l'aspect ratio

`cv2.resize(rgb, (self.load_size, self.load_size))` schiaccia 1536×2048 in
512×512 e poi ristira. Il progetto upstream preserva il rapporto. Correggi con
resize del lato lungo a `load_size` + padding riflesso al multiplo richiesto
dalla rete (8 stadi di downsampling → multiplo di 256), e crop del padding dopo
l'inferenza.

Secondo difetto: `cv2.threshold(..., THRESH_OTSU)` globale su un'istogramma
composto per il ~95 % da bianco sceglie una soglia pessima. Usa una **soglia a
isteresi** (due soglie + connettività, come Canny): mantiene le linee tenui ma
connesse a linee forti e scarta i frammenti tenui isolati.

Terzo: inferenza a 512 poi upscale bicubico a 2048 e *poi* binarizzazione produce
bordi frastagliati. Alza `load_size`, oppure fai tiling con overlap, oppure
binarizza alla risoluzione della rete e fai l'upscale della maschera.

---

## Fase 2 — Nuovo engine classico: flatten → chain → redraw

Questa è la leva più grande sul lato classico, ed è verificata sperimentalmente.

**Diagnosi**: canny/adaptive/xdog reagiscono a *gradienti locali*. Una sorgente
pittorica ha gradienti interni ovunque (pennellate, glow, sfumature) che non sono
confini di oggetto. Serve un passo di **appiattimento che preservi la struttura**
prima della rilevazione, e un passo di **ricostruzione del tratto** dopo.

Aggiungi `opencv-contrib-python>=4.8,<6` come dipendenza (sostituisce
`opencv-python`; è un superset). Sblocca `cv2.ximgproc`: `rollingGuidanceFilter`,
`l0Smooth`, `thinning`, `createEdgeDrawing`, `createStructuredEdgeDetection`,
`guidedFilter`. Verificato disponibile su OpenCV 5.0.0.

### Nuovo engine `engines/chained.py` (nome suggerito: `--style chained`)

Pipeline in tre stadi:

1. **Flatten** — `cv2.ximgproc.rollingGuidanceFilter` (Zhang et al. 2014),
   progettato esattamente per rimuovere strutture a piccola scala preservando i
   bordi a grande scala. Parametri di partenza verificati:
   `sigmaColor=35, sigmaSpace=6, numOfIter=4`.
   Alternativa da valutare: `cv2.ximgproc.l0Smooth(img, lambda_=0.02, kappa=2.0)`
   (L0 gradient minimization, Xu et al. 2011), più aggressivo sulle texture ma
   più lento.

2. **Chain** — `cv2.ximgproc.createEdgeDrawing()`. A differenza di Canny
   restituisce **catene di bordo connesse** (`getSegments()`), non una maschera
   di pixel: risolve alla radice il problema dei contorni spezzati e consente di
   filtrare per lunghezza di percorso. Parametri di partenza verificati:
   `GradientThresholdValue=36, AnchorThresholdValue=8, MinPathLength=30,
   Sigma=1.5, NFAValidation=True`.

3. **Redraw** — per ogni segmento, `cv2.approxPolyDP(seg, epsilon≈1.2, closed=False)`
   per lisciare il tremolio, poi `cv2.polylines(..., thickness, cv2.LINE_AA)` su
   canvas bianco. Questo è ciò che produce il **tratto di larghezza uniforme**
   che distingue il riferimento: non si ridisegna la maschera, si ridisegna la
   *geometria*.

**Risultato misurato del prototipo**: 144 componenti, 11.5 % di inchiostro,
contro le 917 componenti di `cartoon`. Visivamente vicino al riferimento.
Punti deboli residui da affrontare in tuning: rumore residuo su acqua e roccia,
corpi dei dinosauri non chiusi.

### Variante alternativa da implementare e confrontare

`l0Smooth` → Canny a percentile → `cv2.ximgproc.thinning(THINNING_ZHANGSUEN)` →
filtro per lunghezza → `findContours` + `approxPolyDP` + `polylines`.
Misurato: 115 componenti, 8.7 % di inchiostro. Da valutare su più immagini prima
di scegliere quale dei due promuovere a default.

Nota: lo skeletonizzare produce rami spuri sulle biforcazioni. Se il tratto
risulta "peloso", aggiungi un passo di **pruning dei rami** più corti di N pixel
prima del ridisegno.

### Deprecare `adaptive`

A 26 % di copertura e 7515 frammenti non è recuperabile per questo caso d'uso.
Tienilo come stile opzionale documentato come "sketch texturizzato", ma non
presentarlo come opzione per coloring page.

---

## Fase 3 — Sostituire l'engine ML

`Anime2Sketch` è la scelta sbagliata: è addestrato su anime/manga *digitali* a
tratto già netto, non su foto o illustrazioni pittoriche. Richiede inoltre un
download manuale da Google Drive, il che rende l'engine di fatto inutilizzabile.

**Sostituto raccomandato: Informative Drawings** (Chan, Durand, Isola — CVPR
2022, *Learning to generate line drawings that convey geometry and semantics*).
Licenza MIT. È la famiglia di modelli che produce output del tipo mostrato in
`docs/desired.jpg`. Tre checkpoint: `anime_style`, `contour_style`,
`opensketch_style` — per un libro illustrato parti da `contour_style`.

Percorso di integrazione preferito, che evita il download manuale: i pesi sono
ridistribuiti su Hugging Face nel repo `lllyasviel/Annotators` e sono usabili via
`controlnet_aux`:

```python
from controlnet_aux import LineartDetector

detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
line_art = detector(pil_image, coarse=False)
```

**Prima di scrivere codice**: verifica sul repo `huggingface/controlnet_aux` la
firma corrente dell'API e la licenza dei pesi ridistribuiti, e annota l'esito in
`THIRD_PARTY_LICENSES.md` — la skill `add-conversion-style` lo richiede
esplicitamente. Se la licenza dei pesi mirror non è chiara, ripiega
sull'implementazione diretta della rete di `informative-drawings` (il generatore
è piccolo, gira in CPU) con download dal link ufficiale.

Alternative da valutare nello stesso giro, nell'ordine:

- **PidiNet** / **TEED** (edge detector CNN leggeri, entrambi in `controlnet_aux`)
  — più veloci, tratto più "disegnato a mano" di Canny.
- **Sketch Simplification** (Simo-Serra et al., SIGGRAPH 2016/2018) — non un
  estrattore ma un **secondo stadio**: prende line art grezza e rumorosa e
  restituisce tratto pulito e uniforme. Da provare *in cascata* dopo l'engine
  `chained` della Fase 2; è plausibilmente il passo che manca per arrivare alla
  qualità del riferimento.
- **AniLines** (`zhenglinpan/AniLines-Anime-Lineart-Extractor`) — più recente,
  da verificare licenza e peso.

Tieni l'extra `ml` opzionale come ora: il default deve restare zero-ML.

---

## Fase 4 — Strumenti per misurare, senza i quali il tuning è cieco

Niente di quanto sopra è tunabile a occhio. Aggiungi:

1. **`--debug-dir PATH`**: scrive gli stadi intermedi (appiattito, mappa bordi,
   scheletro, ridisegno). Senza questo non si sa quale stadio sta sbagliando.

2. **Modulo `metrics.py`** con metriche calcolabili su un output:
   - copertura di inchiostro (target 5–8 %)
   - numero di componenti connesse (target ~100–200)
   - lunghezza mediana dei tratti
   - frazione di inchiostro in componenti sotto la soglia di lunghezza
     (= indicatore di rumore residuo)

3. **Script `scripts/compare.py`**: esegue tutti gli engine su una cartella di
   immagini e produce una contact sheet affiancata + tabella di metriche.
   Serve a decidere il default su base empirica invece che su una singola foto.

4. **Test di non-regressione**: su immagine sintetica, assertare che ogni engine
   produce copertura di inchiostro dentro una banda. Cattura esattamente il
   fallimento silenzioso di `xdog` (0.05 %) e di `adaptive` (26 %) che oggi
   nessun test rileva.

---

## Ordine di esecuzione e criterio di accettazione

Fase 1 → Fase 4 (metriche) → Fase 2 → Fase 3. Le metriche vanno prima degli
engine nuovi, non dopo.

Criterio: su `docs/starting-image.jpeg`, lo stile di default deve raggiungere
copertura di inchiostro 4–9 % e meno di 250 componenti connesse, con i contorni
dei tre dinosauri e delle due figure in primo piano chiusi e continui.

Commit separati per fase, messaggi conventional-commit, nessun push.
