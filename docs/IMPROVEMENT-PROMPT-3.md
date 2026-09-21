# Prompt per Claude Code — fase 3: bug fix misurabili, poi cambio di direzione

> Incolla tutto il contenuto sotto la riga in una sessione di Claude Code
> aperta nella root di `photo-to-coloring-page`.
>
> Sostituisce i criteri numerici di `IMPROVEMENT-PROMPT.md` e di
> `IMPROVEMENT-PROMPT-2.md`, che erano **entrambi sbagliati** — il primo sui
> target di copertura, il secondo sulla tolleranza della metrica (vedi § 1).
> La diagnosi completa con tutte le misure è in `docs/DIAGNOSIS.md`: leggila
> prima di iniziare.

---

## 0. Regole di ingaggio

- Rispetta `CLAUDE.md`: docstring NumPy in inglese che spiegano *perché*, type
  hints ovunque, test in `tests/` speculari a `src/`, commit conventional-style
  senza alcun trailer di attribuzione AI, **nessun push**.
- **Un commit per fase.** Nel messaggio riporta il numero misurato dopo la
  fase, non "migliora la qualità".
- **Ogni fase ha un criterio di uscita numerico.** Se non lo raggiungi, non
  passare alla fase successiva: fermati e riporta perché.
- I numeri citati in questo documento sono stati misurati il 2026-09-18.
  **Verificali tu stesso prima di usarli come base**, con una eccezione
  importante segnalata in § 3: le misure sull'engine ML sono state fatte con
  una reimplementazione NumPy della rete, non con PyTorch. Le riproduci con
  torch e, se divergono di più del 3 % di F1, fermati e indaga prima di
  procedere.
- Non tarare parametri finché la Fase 0 non è chiusa. La metrica attuale
  premia la direzione sbagliata, e ha già causato due decisioni errate.

---

## Fase 0 — Riparare la metrica (blocca tutto il resto)

### Il problema

`metrics.boundary_f_measure` con `tolerance=2` non misura la qualità del
disegno, misura l'allineamento sub-pixel. Calibrazione, usando come
"predizione" trasformazioni note del riferimento stesso
(`docs/desired.jpg`, normalizzato a 864 px di lato lungo):

| predizione | F1 @1 | F1 @2 | F1 @4 | F1 @8 | inchiostro |
|---|---:|---:|---:|---:|---:|
| riferimento identico | 1.000 | 1.000 | 1.000 | 1.000 | 3.8 % |
| riferimento dilatato 3×3 | — | **1.000** | 1.000 | 1.000 | 11.4 % |
| **riferimento traslato di 3 px** | 0.278 | **0.502** | 0.923 | 0.999 | 3.8 % |
| riferimento traslato di 6 px | 0.190 | 0.308 | 0.546 | 0.973 | 3.7 % |
| rumore casuale 4 % | 0.123 | 0.218 | 0.400 | 0.611 | 4.0 % |
| pagina tutta nera | 0.168 | 0.258 | 0.413 | 0.612 | 100 % |
| `canny` (default attuale) | 0.478 | **0.614** | 0.736 | 0.838 | 11.0 % |

Due conseguenze, entrambe fatali:

1. A tolleranza 2, **il riferimento stesso traslato di 3 px (0.502) perde
   contro `canny` (0.614)**. Il criterio "F1 ≥ 0.75 @tol 2" di
   `IMPROVEMENT-PROMPT-2.md` § 8 chiede di riprodurre le scelte stilistiche di
   MimiPanda entro 2 pixel. Non è un obiettivo ambizioso, è un errore di unità.
2. **Dilatare il riferimento fino all'11 % di inchiostro lascia F1 = 1.000.**
   La metrica non penalizza lo spessore, mentre la recall cresce con il volume
   di inchiostro. È per questo che è stato scelto `nms` come strategia di
   default (vedi Fase 1) e che `canny` all'11 % di inchiostro batte prototipi
   visivamente migliori al 5 %.

A tolleranza 4 px (0.5 % del lato lungo) il floor degenere è 0.41 e un disegno
corretto ma traslato di 3 px prende 0.923: lì c'è dinamica utile fra "rumore" e
"giusto".

### Cosa implementare in `src/coloring_page/metrics.py`

1. **Tolleranza relativa, non assoluta.** Sostituisci il parametro
   `tolerance: int` con una tolleranza derivata dalla dimensione:
   `tolerance_px = max(2, round(0.005 * long_side))` (4 px a 864 px). Tieni
   `tolerance` come override esplicito per i test, ma il default deve essere
   relativo. Aggiorna `compare_boundaries` di conseguenza.

2. **Assottiglia entrambe le mappe a 1 px prima del confronto**, con
   `cv2.ximgproc.thinning`. Senza questo si sta confrontando anche lo spessore
   del pennello, che è una scelta di rendering e non di contenuto — ed è
   esattamente la scappatoia che ha premiato `nms`.

3. **Floor degenere e normalizzazione.** Aggiungi:

   ```python
   def degenerate_floor(gt: np.ndarray, *, tolerance: int) -> float:
       """F1 a cui arriva una predizione priva di informazione.

       Una pagina interamente nera ha recall 1.0 per costruzione, quindi il
       suo F1 non è zero: su `docs/desired.jpg` a tolleranza 4 px vale 0.41.
       Qualunque punteggio va letto rispetto a questo pavimento, non rispetto
       a zero, altrimenti un engine che disegna troppo sembra promettente.
       """
   ```

   e riporta `f1_normalized = (f1 - floor) / (1 - floor)`, clampato a `[0, 1]`.
   Il punteggio che compare nei report e nei commit deve essere quello
   normalizzato.

4. **Vincolo di ammissibilità sull'inchiostro.** La copertura target è **3–7 %**
   (il riferimento sta a 2.2 % con soglia `<128`, 3.8 % con soglia `<160`). Un
   risultato fuori banda non va confrontato per F1: va marcato come non
   ammissibile. Implementalo come campo booleano nel risultato, non come
   penalità dentro l'F1 — mescolare i due rende impossibile capire quale dei
   due sta fallendo.

### Test da aggiungere in `tests/test_metrics.py`

- I baseline degeneri (pagina nera, griglia regolare, rumore uniforme) restano
  **sotto il floor + 0.05** in F1 normalizzato.
- **Il riferimento traslato di 3 px ottiene F1 normalizzato > 0.80.** Questo è
  il test che la metrica attuale fallisce, ed è il motivo per cui esiste questa
  fase. Se non passa, la tolleranza è ancora sbagliata.
- Il riferimento dilatato oltre la banda di inchiostro viene marcato non
  ammissibile.

### Aggiorna anche

- `scripts/compare.py`: colonne `ref_f1_normalized` e `ink_admissible` nel CSV;
  stampa il floor degenere in testa al report.
- `docs/IMPROVEMENT-PROMPT-2.md` § 8: aggiungi in testa una nota che il
  criterio "F1 ≥ 0.75 @tol 2" è ritirato e rimanda a questo documento.

**Criterio di uscita**: i tre test sopra passano, e rieseguendo
`scripts/compare.py` il ranking degli engine cambia rispetto a oggi (se non
cambia, la metrica non è stata effettivamente riparata).

---

## Fase 1 — Bug: `nms_centerline` raddoppia ogni tratto

`postprocess.nms_centerline` applica `cv2.Canny` alla soft map. Canny trova i
**massimi del gradiente**, cioè i due fianchi di un tratto, non la sua cresta.
Un tratto scuro largo 6 px non diventa una linea: diventa il suo contorno.

Riproduzione:

```python
soft = np.full((60, 120), np.uint8(255))
soft[20:26, 10:110] = 20  # tratto spesso 6 px
soft[45:47, 10:110] = 20  # tratto spesso 2 px

nms_centerline(normalize_percentile(soft))
# righe con inchiostro: [19, 20, 21, 22, 23, 24, 25,  44, 45, 46]
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^
#                        i due fianchi del tratto     idem

hysteresis_centerline(normalize_percentile(soft), min_branch_length=3)
# righe con inchiostro: [22, 45]        <- corretto
```

`"nms"` è il default di `soft_map_to_line_art`, quindi il default di
`informative_drawings` **e** di `anime2sketch`. È il motivo per cui l'output
di quegli engine sembra fatto di macchie contornate invece che di linee.

### Cosa fare

1. **Default `strategy="hysteresis"`** in `soft_map_to_line_art`, e nei
   costruttori dei due engine ML.
2. **Rinomina `nms_centerline` in `gradient_edges`.** Non è una centerline, e
   il nome è ciò che ha reso plausibile sceglierla come default. Tienila
   disponibile (è un edge detector legittimo, solo non per questo scopo) con
   una docstring che dice esplicitamente che restituisce i *bordi* di un tratto
   e non il tratto.
3. **Test di non-regressione** sull'esempio sopra: un tratto pieno largo N px
   deve produrre **una sola** riga di inchiostro. Parametrizzalo su
   larghezze 2, 6, 12.
4. Correggi la docstring di `soft_map_to_line_art` che oggi motiva la scelta di
   `nms` con "recupera più inchiostro vero": è vero e irrilevante, raddoppiare
   i tratti raddoppia la recall. Spiega perché quella motivazione era un
   artefatto della metrica.

### Nota minore, stessa famiglia

`redraw_centerline` usa `cv2.findContours`, che traccia il **bordo** di una
maschera. Su uno scheletro da 1 px è innocuo (il contorno percorre la linea
andata e ritorno), ma la funzione è documentata come generica: su una maschera
più spessa raddoppierebbe di nuovo. O documenta il precondizionamento
("accetta solo maschere già assottigliate a 1 px") e aggiungi un assert, o
riscrivila con un tracciamento esplicito della catena. Non lasciarla ambigua:
è la stessa classe di errore della Fase 1 in attesa di ripresentarsi.

**Criterio di uscita**: il test parametrizzato passa, e l'F1 normalizzato di
`informative_drawings` e `anime2sketch` migliora rispetto alla misura di
Fase 0.

---

## Fase 2 — Bug: `informative_drawings` gira a 256×256 quadrati

```python
# src/coloring_page/engines/informative_drawings.py
resized = cv2.resize(rgb, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
```

Tre difetti in una riga:

- **quadrato**: 1536×2048 schiacciato a 256×256 e poi ristirato — la geometria
  esce deformata e i soggetti piccoli (i dinosauri sullo sfondo, le montagne)
  spariscono del tutto;
- **256 invece di 512+**: il preprocessore ufficiale
  (`controlnet_aux.LineartDetector`) usa `detect_resolution=512` applicato al
  **lato corto**, con aspect ratio preservato e arrotondamento a multiplo di
  64;
- **upscale 4–5×** della soft map *prima* della binarizzazione: tratti da 1 px
  diventano gradienti larghi 5 px, cioè i blob che la Fase 1 poi contornava.

`anime2sketch.py` aveva lo stesso bug ed **è già stato corretto**
(`_resize_and_pad`, `load_size=512`). `informative_drawings.py` è rimasto
indietro: è una svista, non una scelta.

### Effetto misurato (`starting-image.jpeg` vs `desired.jpg`, tolleranza grezza)

| configurazione | F1 @2 | F1 @4 | P @4 | R @4 | inchiostro |
|---|---:|---:|---:|---:|---:|
| **attuale** (256², NMS) | **0.294** | 0.468 | — | — | 17.9 % |
| 512 lato corto + hysteresis | 0.392 | 0.620 | 0.623 | 0.617 | 7.4 % |
| 768 lato corto + hysteresis | 0.488 | 0.685 | 0.697 | 0.674 | 8.5 % |
| **1024 lato corto + hysteresis** | **0.536** | **0.713** | 0.730 | 0.696 | 8.6 % |

**+82 % di F1 senza toccare la rete né i pesi**, e il numero cresce ancora con
la risoluzione: la rete è completamente convoluzionale, la costante 256 non ha
alcuna giustificazione.

> ⚠️ **Queste righe sono state misurate con una reimplementazione NumPy del
> generatore**, perché l'ambiente di diagnosi non aveva PyTorch. L'architettura
> è la stessa (`Generator(3, 1, 3)`, InstanceNorm, ReflectionPad) e i pesi sono
> quelli del repo, ma **riesegui la tabella con torch prima di considerarla
> acquisita**. Se diverge di più del 3 % di F1, fermati: significa che
> `_informative_drawings_arch.py` e il checkpoint non combaciano come
> pensiamo.

### Cosa fare

1. Sostituisci il resize quadrato con uno stile ControlNet:

   ```python
   def _resize_short_side(image: np.ndarray, resolution: int, *, multiple: int = 64) -> np.ndarray:
       """Scale so the SHORT side is ``resolution``, rounded to a multiple of 64.

       Matches ``controlnet_aux.util.resize_image``, which is what the
       redistributed weights were exercised with: a fully-convolutional
       generator has no fixed input size, but it does have a scale it was
       trained at, and feeding it a squashed square changes both the aspect
       ratio and the apparent scale of every object in the scene.
       """
   ```

   Verifica la firma corrente su
   `https://github.com/huggingface/controlnet_aux` (file
   `src/controlnet_aux/lineart/__init__.py`) prima di scrivere: quel file è la
   fonte autorevole per il preprocessing, non la documentazione.

2. `load_size` → `detect_resolution`, default **1024**, documentato come "lato
   corto" e non come "lato del quadrato". Deprecane il vecchio nome se
   qualcosa lo usa.

3. **Nessun upscale della soft map prima della binarizzazione.** Fai
   l'inferenza alla risoluzione di lavoro e binarizza lì; se serve un resize
   finale, applicalo alla line art già binarizzata e ridisegnata.

4. Misura anche a `detect_resolution` 1280 e 1536 e riporta dove la curva si
   appiattisce. Il default deve essere il ginocchio della curva, non il massimo
   (il costo cresce quadraticamente).

5. Controlla la **polarità** con un test esplicito. L'output `Sigmoid` della
   rete ha `0 = inchiostro`, e `controlnet_aux` inverte alla fine (`255 -
   detected_map`) solo perché ControlNet vuole linee bianche su nero. La
   convenzione del progetto è la stessa della rete, quindi **non serve
   invertire** — ma è un'assunzione che merita un test, non un commento.

**Criterio di uscita**: F1 normalizzato di `informative_drawings` almeno
raddoppiato rispetto alla misura di Fase 0, con inchiostro dentro la banda
3–7 %.

---

## Fase 3 — Checkpoint: probabilmente è quello sbagliato

Il file in `weights/informative_drawings.pth` produce uno stile "matita", con
tratteggio e ombreggiatura: è il checkpoint **fine** (`sk_model.pth`). Il
tratteggio è precisamente ciò che stiamo cercando di eliminare, e infatti la
precision resta bassa anche dopo la Fase 2.

`scripts/fetch_weights.py` sa già scaricare il coarse ma non è stato usato:

```bash
python scripts/fetch_weights.py --filename sk_model2.pth \
    --dest weights/informative_drawings-coarse.pth
```

- Misura **fine vs coarse** con la metrica della Fase 0 su tutte e tre le
  coppie di riferimento. Non assumere: carica entrambi e decidi sui numeri.
- Pinna lo SHA256 del vincitore in `fetch_weights.py` (oggi lo stampa soltanto).
- Aggiorna `THIRD_PARTY_LICENSES.md` con l'esito della verifica di licenza del
  mirror `lllyasviel/Annotators`, come richiesto dalla skill
  `add-conversion-style`.
- Riscrivi il messaggio di errore dei pesi mancanti perché nomini il checkpoint
  effettivamente scelto.

*Confidenza: alta sul fatto che il checkpoint attuale sia il fine; media su
quanto guadagno dia il coarse.*

**Criterio di uscita**: una tabella fine-vs-coarse nel messaggio di commit, e
il default del repo che punta al vincitore.

---

## Fase 4 — Nuovo engine `region`: confini di regione, non bordi di gradiente

Questa è la parte che cambia direzione, ed è l'unica leva rimasta sul lato
classico.

### Il ragionamento

Un edge detector chiede: *"qui c'è un salto di intensità?"*. Un disegno da
colorare risponde a una domanda diversa: *"qui finisce un oggetto e ne comincia
un altro?"*. Su una tavola illustrata le due domande hanno risposte diverse
ovunque, perché glow, sfumature e riflessi producono salti di intensità che non
sono confini di oggetto. `IMPROVEMENT-PROMPT-2.md` § 2.1 lo ha già dimostrato
sperimentalmente (36 combinazioni di parametri, nessuna sopra F1 0.587) e ne
ha tratto la conclusione giusta — *serve semantica* — ma la direzione presa
poi (gating con un modello ML) attacca il problema dal lato sbagliato: il
gating può solo **togliere** linee da un insieme di candidate già sbagliate.

Segmentare in regioni e disegnarne i **confini** produce contorni chiusi per
costruzione, e non produce nulla dove c'è solo un'ombra.

### Prototipo già misurato

`ximgproc.l0Smooth(λ=0.02, κ=2.0)` → `cv2.pyrMeanShiftFiltering(sp=14, sr=20,
maxLevel=2)` → gradiente calcolato in spazio **Lab** (non su grayscale: due
colori diversi con la stessa luminanza sono un confine di oggetto e un
grayscale li perde) → soglia al 94° percentile → `ximgproc.thinning` →
`approxPolyDP` + `polylines`.

| immagine | engine | F1 @4 | P @4 | R @4 | inchiostro |
|---|---|---:|---:|---:|---:|
| `starting-image` | `canny` | 0.736 | 0.678 | 0.805 | 9.8 % |
| | `chained` | 0.731 | 0.683 | 0.786 | 10.9 % |
| | **region** | 0.681 | **0.789** | 0.599 | **5.3 %** |
| `starting-image-5` | `canny` | 0.861 | 0.892 | 0.832 | 8.6 % |
| | **region** | 0.789 | **0.942** | 0.679 | **5.9 %** |
| `starting-image-7` | `chained` | 0.619 | 0.494 | 0.829 | 14.4 % |
| | **region** | 0.610 | **0.604** | 0.616 | **5.5 %** |

Stesso F1 grezzo con **metà dell'inchiostro** e precision sistematicamente più
alta: quasi tutto quello che disegna è giusto, ed è dentro la banda di
ammissibilità mentre `canny` e `chained` ne sono fuori. Visivamente è il
prototipo più vicino al riferimento fra tutti quelli provati — contorni dei
dinosauri chiusi, dinosauri piccoli e montagne presenti, acqua e cielo puliti.
Confronta tu stesso con `docs/diagnosis-contact-sheet.png`.

Che l'F1 grezzo non salga è la conferma del § 0: quella metrica premia la
recall. Con la metrica della Fase 0 il confronto va rifatto e il ranking
dovrebbe cambiare.

### Cosa implementare

Segui `.claude/skills/add-conversion-style/SKILL.md` per intero (modulo in
`engines/`, registrazione in `registry.py`, test in `tests/engines/`).

Struttura a stadi espliciti, ognuno salvato via `DebugSink`:

1. `flatten` — `l0Smooth`, parametri esposti nel costruttore;
2. `segment` — `pyrMeanShiftFiltering`; **valuta come alternativa**
   `ximgproc.createSuperpixelSEEDS` o SLIC + region merging su distanza Lab,
   che danno controllo esplicito sul numero di regioni finali invece che
   indiretto via `sr`;
3. `boundaries` — gradiente Lab + soglia a percentile. **Valuta in alternativa
   di tracciare direttamente i bordi della mappa di etichette** con
   `findContours`: dà contorni chiusi senza passare da una soglia, che è il
   punto dell'intero approccio;
4. `filter` — scarta i confini per **lunghezza del contorno** e per **contrasto
   medio fra le due regioni adiacenti**. Quest'ultimo è il criterio che un edge
   detector non può avere e il motivo per cui questo approccio è diverso, non
   solo un'altra taratura: un confine fra due regioni quasi identiche è
   un artefatto di segmentazione, non un bordo;
5. `redraw` — riusa `postprocess.redraw_segments`.

Deriva ogni kernel in pixel da `pipeline.derive_kernel_size`, non hardcodarlo:
gli engine assumono la working resolution (`DEFAULT_WORKING_DIMENSION = 1400`).

Misura le tre varianti dello stadio 2 e le due dello stadio 3 con la metrica
della Fase 0 e scegli sui numeri, documentando l'esito nella docstring della
classe.

**Criterio di uscita**: `region` batte `canny` e `chained` in F1 normalizzato
su almeno 2 delle 3 coppie, restando dentro la banda di inchiostro su tutte e
tre.

---

## Fase 5 — Cambiare il default e allineare la documentazione

`--style` di default è `canny` (`cli.py:56`). È l'engine che risolve il
problema sbagliato nel modo più diretto: va sostituito.

- Scegli il default fra `region` (zero ML) e `informative_drawings` corretto
  (richiede l'extra `ml`) **eseguendo `scripts/compare.py`**, non a occhio. Il
  default del repo non può richiedere l'extra `ml`, quindi se vince
  `informative_drawings` il default zero-ML resta `region` e la README deve
  dire chiaramente quando conviene installare l'extra.
- Aggiorna la sezione "Conversion styles" della README con i numeri nuovi e
  rimuovi le raccomandazioni basate su misure ritirate.
- `adaptive` e `xdog`: o li ripari o li togli dai `--style` presentati come
  opzioni per coloring page. `adaptive` al 47 % di inchiostro non è recuperabile
  per questo caso d'uso; documentalo come "sketch texturizzato" e basta.
- Rigenera `outputTests/` con gli engine aggiornati, così la cartella smette di
  documentare uno stato che non esiste più.

---

## Fase 6 — Solo dopo le precedenti: la famiglia di modelli giusta

Non iniziare questa fase finché le Fasi 0–5 non sono chiuse e misurate.

`desired*.jpg` ha la firma della famiglia **lineart anime / manga line
extraction**: contorno chiuso per oggetto, spessore uniforme, zero
ombreggiatura, dettagli reinterpretati (l'erba diventa trattini stilizzati, i
volti hanno occhi e bocca ridisegnati). `informative-drawings` in stile
`contour`/`coarse` è la cosa più vicina che abbiamo già in casa, ma non è
la stessa famiglia.

Da valutare, in ordine, ognuno con verifica di licenza **prima** di scrivere
codice (requisito della skill `add-conversion-style`):

1. **`LineartAnimeDetector`** di `controlnet_aux` (checkpoint `netG.pth` sullo
   stesso mirror `lllyasviel/Annotators`). Architettura diversa da quella già
   vendorizzata: serve un nuovo modulo `_arch`.
2. **MangaLineExtraction** (Li et al.) — stessa famiglia, verifica licenza.
3. **Sketch Simplification** (Simo-Serra et al., SIGGRAPH 2016/2018) — non un
   estrattore ma un **secondo stadio**: prende line art grezza e restituisce
   tratto pulito e uniforme. Va applicato **in cascata** sul miglior engine
   precedente. È plausibilmente ciò che dà al riferimento il suo aspetto
   "disegnato", ed è l'unica cosa in questo elenco che può alzare la precision
   senza abbassare la recall.

L'extra `ml` resta opzionale: il percorso di default deve restare zero-ML.

---

## Cosa NON fare

- **Non tarare i parametri degli engine classici esistenti.** Il tetto è già
  stato misurato (F1 0.587 su 36 combinazioni) e la causa è strutturale, non
  parametrica.
- **Non combinare engine classici fra loro.** Sbagliano tutti nello stesso
  modo, quindi la combinazione non aggiunge informazione.
- **Non reintrodurre il gating ML come leva principale.** È già stato provato e
  misurato: alza la precision a 0.85 facendo crollare la recall a 0.47. Può
  solo togliere linee da un insieme di candidate già sbagliate.
- **Non ottimizzare l'F1 grezzo.** Dopo la Fase 0 il numero da riportare è
  quello normalizzato, con il vincolo di inchiostro. Un F1 grezzo che sale
  mentre l'inchiostro esce dalla banda è un peggioramento travestito — è
  esattamente come siamo arrivati qui.

---

## Sul set di valutazione

3 riferimenti per 7 input, e le coppie 5 e 7 hanno aspect ratio diverso
dall'input (MimiPanda ha ritagliato: `starting-image-5` è 1536×1292, ratio
1.19; `desired-5` è 1152×864, ratio 1.33). Il codice segnala già il mismatch
con un warning, ma i numeri su quelle due coppie vanno letti come approssimati.

Prima della Fase 4, se possibile: genera con MimiPanda i riferimenti per tutte
e 7 le immagini e tieni 2 coppie fuori dal tuning come validation set. Con 3
coppie, di cui 2 disallineate, il sovradattamento è garantito — e nessuna delle
metriche di questo documento se ne accorgerebbe.
