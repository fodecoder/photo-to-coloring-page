# Prompt di riscrittura per Claude Code — da edge map a pipeline di produzione

Contesto della decisione (già presa, non rimetterla in discussione nei prompt):

- **ML permesso** in produzione, purché con licenza permissiva (Apache-2.0 / MIT) e
  peso verificabile via checksum.
- **Deliverable finale: vettoriale** (SVG + PDF pronto da stampare). Il PNG diventa
  un export secondario ottenuto rasterizzando il vettoriale, non il formato interno.

Eseguire i prompt **in ordine**. Ognuno è autonomo e finisce con un criterio di
uscita verificabile. Non passare al successivo se il precedente non lo soddisfa.

---

## Prompt 1 — Spike di validazione (throwaway, non entra in `src/`)

```text
Obiettivo: verificare, PRIMA di riscrivere la pipeline, che una partizione in
regioni ottenuta per segmentazione batta l'approccio a gradiente che il repo usa
oggi. Codice usa-e-getta in scripts/spike_segmentation.py, non in src/.

Contesto: gli engine attuali (canny, chained, region, informative_drawings,
anime2sketch) rispondono tutti al gradiente locale o a un modello di line
extraction generico. Vedi docs/DIAGNOSIS.md. Il risultato non è utilizzabile:
tratti interrotti, contorni non chiusi, texture di sfondo trasformata in rumore.

Fai questo:
1. Aggiungi l'extra opzionale `seg` in pyproject.toml con `segment-anything-2`
   (Apache-2.0) e `huggingface_hub`. Estendi scripts/fetch_weights.py per
   scaricare il checkpoint SAM 2.1 Hiera-small verificando lo SHA256.
2. Per ognuna delle 7 immagini in docs/starting-image*.jpeg:
   a. genera le maschere con l'automatic mask generator di SAM 2;
   b. risolvi le sovrapposizioni in una LABEL MAP intera, senza buchi e senza
      overlap (assegna ogni pixel alla maschera più piccola che lo contiene;
      i pixel non coperti vanno alla regione vicina più simile in spazio Lab);
   c. scarta le regioni con area < 0.15% dell'area immagine fondendole nella
      vicina con ΔE Lab minore, iterando fino a convergenza;
   d. disegna i confini della label map con cv2.findContours su ogni etichetta.
3. Produci una contact sheet docs/spike-contact-sheet.png: input | desired |
   region attuale | partizione SAM, per le 3 immagini che hanno un riferimento.
4. Riporta per ogni immagine: numero di regioni finali, copertura di inchiostro %,
   numero di endpoint penzolanti (pixel di scheletro con esattamente 1 vicino).

Criterio di uscita — riportalo esplicitamente come PASS/FAIL:
- copertura di inchiostro tra il 3% e l'8%;
- endpoint penzolanti = 0 (i confini di una label map sono chiusi per costruzione:
  se non lo sono, c'è un bug nel passo 2b);
- numero di regioni tra 40 e 400 per immagine.
Non usare boundary_f_measure per decidere: docs/DIAGNOSIS.md §4 dimostra che quella
metrica premia lo spessore e la recall. Decidi sui tre numeri sopra + la contact
sheet.

Se FAIL, fermati e scrivi in docs/ perché, senza tentare taratura a caso.
```

---

## Prompt 2 — Cambio di rappresentazione interna: `Drawing`, non `np.ndarray`

Questo è il prompt che vale di più. Tutto il resto dipende da qui.

```text
Obiettivo: la pipeline oggi passa array di pixel fra gli stadi e produce una
bitmap binaria. È la causa strutturale del fatto che l'output non sia
utilizzabile: lo spessore del tratto dipende dalla risoluzione, i contorni non
sono chiusi per costruzione, e non esiste modo di filtrare "questa regione è
troppo piccola per essere colorata a pennarello".

Introduci un modello di dominio vettoriale e rifonda ConversionEngine su di esso.

1. Nuovo modulo src/coloring_page/drawing.py:
   - `@dataclass(frozen=True) class Path`: `points: np.ndarray` (N,2) float32 in
     coordinate NORMALIZZATE [0,1] rispetto al lato lungo, `closed: bool`,
     `kind: Literal["boundary", "detail"]`, `source_area: float | None` (area
     normalizzata della regione più piccola adiacente, serve al filtro di
     salienza a valle).
   - `@dataclass(frozen=True) class Drawing`: `paths: tuple[Path, ...]`,
     `aspect_ratio: float`, `meta: Mapping[str, object]`.
     Metodi: `filter(predicate)`, `simplify(epsilon)`, `bounds()`.
   - Le coordinate sono normalizzate perché la risoluzione di lavoro
     dell'engine e la risoluzione di stampa sono due cose diverse e finora sono
     state confuse. Documenta questo nel docstring del modulo.

2. `ConversionEngine.convert` cambia firma: da `(np.ndarray) -> np.ndarray` a
   `(np.ndarray) -> Drawing`. Aggiorna engines/base.py, tutti gli engine in
   engines/, registry.py, pipeline.py, cli.py e i test.

3. Gli engine classici esistenti (canny, adaptive, xdog, cartoon, chained,
   skeleton) vanno adattati con un adattatore condiviso raster→vettore in
   drawing.py: `paths_from_mask(mask: np.ndarray) -> tuple[Path, ...]` che fa
   thinning (cv2.ximgproc.thinning) + tracciamento esplicito dei rami dello
   scheletro. NON usare cv2.findContours su una maschera spessa: traccia il
   bordo del tratto e raddoppia ogni linea (stesso errore già documentato in
   docs/DIAGNOSIS.md §2 per nms_centerline).
   Questi engine restano nel repo come baseline di confronto ma NON sono più
   candidati al default: marcali `experimental = True` sulla classe e
   nascondili da --help a meno di --show-experimental.

4. Test: per ogni engine registrato, un test parametrizzato che su un'immagine
   sintetica verifica che il Drawing restituito abbia paths non vuoti, tutti i
   punti in [0,1], e nessun path con meno di 2 punti.

Vincolo: nessuna regressione di mypy --strict e ruff check.
```

---

## Prompt 3 — L'engine di produzione

```text
Obiettivo: implementare src/coloring_page/engines/lineart.py, l'engine
`--style lineart` destinato a diventare il default. Presuppone che il Drawing di
drawing.py esista (Prompt 2) e che lo spike del Prompt 1 sia PASS.

Architettura a quattro stadi. Ogni stadio è una funzione pura e testabile a sé,
non un metodo privato di 200 righe.

Stadio A — partizione semantica (`_segment`)
  SAM 2.1 automatic mask generator → label map intera senza overlap né buchi
  (promuovi il codice dello spike). Esegui alla risoluzione di lavoro, non a
  risoluzione fissa: lato corto arrotondato a multiplo di 64, aspect ratio
  preservato.

Stadio B — linee di dettaglio interne (`_detail_lines`)
  Un modello di line extraction per ciò che la segmentazione non vede (occhi,
  bocche, pieghe, pattern voluti). Candidati, in ordine di preferenza, con
  licenza da VERIFICARE e trascrivere in THIRD_PARTY_LICENSES.md prima dell'uso:
  MangaLineExtraction_PyTorch (ljsabc, MIT) e lineart_anime di controlnet_aux.
  NON riusare il checkpoint fine di Informative Drawings già in weights/: produce
  tratteggio e ombreggiatura, è lo stile sbagliato (docs/DIAGNOSIS.md §3).
  Binarizza con hysteresis_centerline, mai con la strategia "nms".

Stadio C — fusione e filtro di salienza (`_merge_and_prune`)
  È QUI che si decide se l'output è colorabile. Un confine di regione entra nel
  Drawing solo se supera ALMENO uno di:
    - l'area della regione più piccola che separa, riportata alla dimensione di
      stampa, è >= `min_region_area_mm2`;
    - il ΔE Lab medio fra le due regioni adiacenti è >= `min_region_contrast`;
    - proviene dallo stadio B con risposta sopra soglia.
  Le linee di dettaglio (B) che cadono dentro una regione scartata vanno scartate
  con lei. Questo è il meccanismo che elimina erba, grana della carta e scaffali
  di sfondo senza toccare i soggetti.

Stadio D — regolarizzazione (`_regularize`)
  Semplificazione (approxPolyDP con epsilon relativo al lato lungo), poi
  smoothing Chaikin o fitting a Bézier cubica. Il risultato deve essere una
  curva morbida: il riferimento docs/desired*.jpg non ha scalettature, l'output
  attuale sì. Elimina i path più corti di `min_path_length_mm` e i path
  duplicati (stesso percorso trovato sia da A che da B) con una soglia di
  distanza di Hausdorff.

Parametrizzazione: i parametri dell'engine sono espressi in MILLIMETRI alla
dimensione di stampa, non in pixel. `min_region_area_mm2`, `min_path_length_mm`.
Motivo: "quanto piccola può essere un'area perché un bambino la colori" è una
grandezza fisica, indipendente dalla risoluzione dell'input. Ogni parametro in
pixel nella firma pubblica di questo engine è un bug di design.

Device: parametro esplicito `device: Literal["auto","cpu","cuda","mps"]`, default
"auto". Se torch non è installato o i pesi mancano, solleva un errore tipizzato
con istruzioni, non un ImportError nudo.

Test: usa immagini sintetiche (numpy/PIL) e monkeypatch dei modelli — nessun test
deve scaricare pesi in CI. Marca @pytest.mark.slow gli eventuali test con pesi
reali e escludili dal default con addopts in pyproject.toml.
```

---

## Prompt 4 — Rendering, export e garanzia di colorabilità

```text
Obiettivo: trasformare un Drawing in un artefatto stampabile e DIMOSTRARE che è
colorabile. Oggi non esiste alcun controllo del fatto che i contorni siano
chiusi: è la ragione principale per cui l'output è inutilizzabile in produzione.

1. src/coloring_page/page.py — `@dataclass(frozen=True) class PageSpec`:
   `size: Literal["A4","A5","LETTER"] | tuple[float,float]` in mm,
   `orientation`, `margin_mm`, `dpi: int = 300`, `stroke_width_mm: float = 0.7`.
   Il contenuto viene inscritto nell'area utile preservando l'aspect ratio.

2. src/coloring_page/render.py:
   - `to_svg(drawing, spec) -> str`: un <path> per Path, stroke in mm,
     `stroke-linecap="round"`, `stroke-linejoin="round"`, `fill="none"`,
     `shape-rendering="geometricPrecision"`. Nessuna dipendenza esterna: è
     generazione di testo XML, usa xml.etree con escaping corretto.
   - `to_pdf(drawing, spec) -> bytes`: vettoriale vero, non un PNG incorporato.
     Valuta reportlab (BSD) o la scrittura diretta del content stream; motiva la
     scelta in un commento.
   - `to_png(drawing, spec) -> np.ndarray`: rasterizzazione con antialiasing a
     `spec.dpi`, per anteprima e per la validazione del punto 3.

3. src/coloring_page/validate.py — `validate(drawing, spec) -> QualityReport`.
   Rasterizza a spec.dpi e misura:
   - `ink_coverage`: frazione di pixel inchiostrati (atteso 0.03–0.08);
   - `enclosed_regions`: flood fill dal bordo esterno, poi connected components
     sul complemento; è il numero di aree effettivamente colorabili;
   - `leaking_regions`: regioni che il flood fill esterno raggiunge attraverso un
     contorno aperto — deve essere 0 o quasi;
   - `min_region_area_mm2` e il percentile 5;
   - `dangling_endpoints`: pixel di scheletro con un solo vicino.
   `QualityReport.passed` applica soglie configurabili. La CLI esce con codice 1
   se il report fallisce e --strict è attivo.

   Questa sostituisce boundary_f_measure come criterio di accettazione. La
   metrica vecchia confronta con un riferimento che è una REINTERPRETAZIONE
   artistica e, come dimostra docs/DIAGNOSIS.md §4, premia chi disegna di più.
   Sposta metrics.py sotto scripts/ come strumento diagnostico e togli ogni
   riferimento ad essa dai criteri di accettazione nel README e nei doc.

4. Test con figure sintetiche a verità nota: un cerchio (1 regione racchiusa,
   0 leaking), un cerchio con un varco di 5 px (leaking > 0), due rettangoli
   adiacenti (2 regioni). Sono i test che avrebbero intercettato il problema
   fin dall'inizio.
```

---

## Prompt 5 — Superficie pubblica, robustezza, riproducibilità

```text
Obiettivo: rendere il pacchetto utilizzabile da un servizio, non solo da una
shell. Oggi l'unica API è la CLI e ogni errore arriva all'utente come stack trace.

1. API pubblica in src/coloring_page/__init__.py:
     convert_image(src: Path | np.ndarray, *, profile: Profile) -> Result
   dove `Result` espone `drawing`, `report: QualityReport`, `timings: dict[str,float]`
   e i metodi `save_svg/save_pdf/save_png`. La CLI diventa un wrapper sottile su
   questa funzione. Nessuna I/O di file dentro gli engine.

2. `Profile` (dataclass frozen, serializzabile da/verso JSON): `page: PageSpec`,
   `style: str`, `detail: Literal["toddler","child","adult"]`, `device`, `seed`.
   `detail` mappa sui parametri in mm dell'engine: toddler = poche regioni grandi
   (min_region_area_mm2 alto), adult = molti dettagli. È il vero parametro di
   prodotto; `--thickness` in pixel va deprecato.
   Aggiungi `--profile FILE.json` alla CLI.

3. Errori tipizzati in exceptions.py, tutti derivati da `ColoringPageError`:
   `UnsupportedImageError`, `ImageTooLargeError`, `WeightsMissingError`,
   `WeightsChecksumError`, `EngineUnavailableError`, `QualityGateError`.
   La CLI li cattura e stampa un messaggio a una riga su stderr + exit code
   dedicato; lo stack trace solo con --verbose.

4. Limiti di risorse: `max_pixels` (default ~40 MP) con errore esplicito sopra
   soglia, e nessuna allocazione intermedia non necessaria — profila il picco di
   RSS su un input 6000x4000 e riportalo nel README.

5. Riproducibilità: seed propagato a numpy e torch, `torch.use_deterministic_algorithms`
   dove possibile, versioni pinnate con lower E upper bound, SHA256 di OGNI
   checkpoint verificato al caricamento (fallisci, non avvisare). Directory
   cache dei pesi da variabile d'ambiente `COLORING_PAGE_WEIGHTS_DIR`,
   documentata in .env.example; modalità offline se i pesi ci sono già.

6. Logging con il modulo `logging` (mai print fuori dalla CLI), un log strutturato
   per stadio con durata. `--debug-dir` che scrive gli intermedi di ogni stadio
   (label map colorata, maschera di dettaglio, drawing pre e post filtro).

7. Batch: parallelizza su ProcessPoolExecutor per gli engine CPU e mantieni
   seriale il percorso GPU (un solo modello in VRAM); `--jobs` con default
   sensato. Un fallimento su un file non deve interrompere il batch: raccogli e
   riporta un sommario finale.

8. README riscritto: nuovo default, tabella dei profili, sezione licenze dei
   modelli, requisiti hardware, e i numeri di QualityReport sulle 7 immagini di
   riferimento al posto delle tabelle di F1.
```

---

## Cosa NON fare (metterlo nei prompt se Claude Code devia)

- Non tarare soglie sulle 3 coppie input/desired disponibili: 3 esempi, di cui 2
  con aspect ratio disallineato (docs/DIAGNOSIS.md §7). Qualsiasi numero ottenuto
  così è sovradattamento.
- Non aggiungere un ennesimo engine a gradiente sperando in una taratura migliore.
  Lo sweep è già stato fatto due volte e il tetto è quello che si vede in
  `outputTests/`.
- Non usare `cv2.findContours` su maschere più spesse di 1 px.
- Non reintrodurre `boundary_f_measure` come criterio di merge.
