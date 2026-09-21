# Prompt per Claude Code — fase 2: selezione semantica e metrica reale

> Incolla tutto il contenuto sotto la riga in una sessione di Claude Code aperta
> nella root di `photo-to-coloring-page`. Sostituisce gli obiettivi numerici di
> `docs/IMPROVEMENT-PROMPT.md`, che erano sbagliati (vedi § 0).

---

## 0. Correzione degli obiettivi del prompt precedente

I target del prompt precedente ("copertura inchiostro 4–9 %, meno di 250
componenti connesse") erano derivati da una lettura errata di `docs/desired.jpg`.
Misura corretta del riferimento, normalizzato a 864 px di lato lungo:
**copertura inchiostro 4.37 %**, e il conteggio di componenti connesse è inutile
come metrica perché sul riferimento JPEG è gonfiato a 2568 dagli artefatti di
compressione. Non usare più il conteggio di componenti come criterio.

Serve una metrica di similarità vera, che al momento non esiste nel repo.

---

## 1. Implementare la metrica prima di toccare gli engine

Estendi `metrics.py` con la **boundary F-measure** in stile BSDS500 (Martin et
al.), che è lo standard per confrontare mappe di bordi: si abbinano i pixel di
inchiostro predetti a quelli del riferimento entro una tolleranza spaziale.
Implementazione economica ed equivalente al primo ordine, via distance transform:

```python
def boundary_f_measure(
    pred: np.ndarray, gt: np.ndarray, *, tolerance: int = 2
) -> tuple[float, float, float]:
    """Precision/recall/F1 of predicted strokes against a reference drawing.

    Pixel-wise comparison is meaningless for line art: two drawings of the
    same scene never place strokes on identical pixels. The standard fix
    (BSDS500 boundary benchmark) is to count a predicted stroke pixel as
    correct when a reference stroke pixel lies within `tolerance`, and vice
    versa for recall. A distance transform gives this without the full
    bipartite matching of the original benchmark.
    """
    pred_ink = (pred < 160).astype(np.uint8)
    gt_ink = (gt < 160).astype(np.uint8)
    dist_to_gt = cv2.distanceTransform(1 - gt_ink, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(1 - pred_ink, cv2.DIST_L2, 3)
    precision = float((dist_to_gt[pred_ink > 0] <= tolerance).mean()) if pred_ink.any() else 0.0
    recall = float((dist_to_pred[gt_ink > 0] <= tolerance).mean()) if gt_ink.any() else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
```

**Vincoli non negoziabili sulla metrica**, perché ho verificato che sbagliarli la
rende attivamente dannosa:

- **Normalizza entrambe le immagini allo stesso lato lungo** (864 px) prima di
  confrontarle. Senza questo la metrica confronta risoluzioni diverse.
- **Usa `tolerance = 2` px come metrica primaria.** Ho misurato che a
  tolleranza 11 px (1 % della diagonale, valore usato in letteratura su foto
  naturali) **un'immagine completamente nera ottiene F1 = 0.670**, più di
  `anime2sketch` a 0.747 ma vicinissima, e `adaptive` — visivamente inutile, 47 %
  di inchiostro — ottiene 0.780. Una tolleranza larga premia il disegnare *più*
  linee, che è esattamente la direzione sbagliata.
- **Aggiungi i baseline degeneri al test suite**: immagine tutta nera e griglia
  regolare a 8 px. A tolleranza 2 px ottengono F1 ≈ 0.26. Qualunque engine che
  non li batte di un margine ampio è rumore travestito. Questo test impedisce di
  "ottimizzare" la metrica riempiendo la pagina.

Aggiungi il calcolo di precision/recall/F1 a `scripts/compare.py`, con le coppie
input/riferimento disponibili (`starting-image.jpeg` ↔ `desired.jpg`,
`starting-image-5.jpeg` ↔ `desired-5.jpg`, `starting-image-7.jpeg` ↔
`desired-7.jpg`; verifica prima gli aspect ratio, le coppie 5 e 7 non combaciano
e potrebbero essere crop diversi).

---

## 2. Stato reale: misure e diagnosi

Con la metrica sopra, a tolleranza 2 px, su `starting-image.jpeg` vs
`docs/desired.jpg` (inchiostro di riferimento 4.37 %):

| engine | precision | recall | **F1** | inchiostro |
|---|---:|---:|---:|---:|
| canny | 0.526 | 0.728 | **0.611** | 14.70 % |
| skeleton | 0.545 | 0.673 | **0.602** | 15.86 % |
| chained | 0.502 | 0.711 | **0.589** | 16.99 % |
| cartoon | 0.435 | 0.679 | **0.530** | 19.53 % |
| anime2sketch | 0.297 | 0.519 | **0.378** | 20.18 % |
| xdog | 0.331 | 0.346 | **0.338** | 10.44 % |
| adaptive | 0.197 | 0.912 | **0.324** | 47.52 % |
| *tutto nero (degenere)* | *0.154* | *1.000* | *0.267* | *100 %* |

**Diagnosi: il problema è la precision, non la recall.** La recall è già
0.67–0.73 — le linee che servono ci sono quasi tutte. La precision è 0.50–0.55:
**circa metà dell'inchiostro disegnato non esiste nel riferimento.** La copertura
di inchiostro è 15–20 % contro il 4.37 % del target, cioè 3–4 volte troppo.

Gli engine non devono aggiungere linee. Devono togliere linee.

### 2.1 Il tetto dell'approccio classico è ≈ 0.59, verificato

Ho fatto uno sweep su `chained` (thickness ∈ {1,2}, `GradientThresholdValue` ∈
{36,60,90}, `MinPathLength` ∈ {30,60,100}, due set di parametri di rolling
guidance — 36 combinazioni). **Nessuna combinazione supera F1 = 0.587.**

Il profilo del sweep è la cosa importante:

| thickness | gradient thr | min len | precision | recall | F1 | inchiostro |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 30 | 0.531 | 0.656 | 0.587 | 12.36 % |
| 1 | 36 | 30 | 0.569 | 0.579 | 0.574 | 5.09 % |
| 1 | 60 | 30 | 0.652 | 0.433 | 0.520 | 3.16 % |
| 1 | 60 | 60 | 0.677 | 0.333 | 0.446 | 2.21 % |

Rendendo l'engine più selettivo la precision sale (0.53 → 0.68) ma **la recall
crolla** (0.66 → 0.33): scarta le linee sbagliate. Toglie il contorno del
dinosauro insieme all'increspatura dell'acqua, perché hanno la stessa magnitudine
di gradiente. Non è un problema di parametri, è un problema **semantico**: una
soglia sul gradiente non può sapere cosa sia un oggetto.

**Conseguenza operativa: smetti di tarare gli engine classici.** Non combinare
engine classici fra loro — sbagliano tutti nello stesso modo, quindi la
combinazione non aggiunge informazione. Serve un modello che porti semantica.

---

## 3. Sbloccare `informative_drawings` (priorità assoluta)

L'engine è implementato e l'architettura vendorizzata in
`_informative_drawings_arch.py` è corretta (`Generator(input_nc=3, output_nc=1,
n_residual_blocks=3)`, ResNet con InstanceNorm). Manca solo il file di pesi, e il
messaggio d'errore manda su Google Drive, che è un vicolo cieco per
l'automazione.

I pesi sono ridistribuiti su Hugging Face, scaricabili senza login:

```
https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth
https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth
```

Sono i pesi del preprocessore "lineart" di ControlNet, che usa esattamente questo
`Generator(3,1,3)`. `sk_model.pth` è la variante a dettaglio fine,
`sk_model2.pth` quella *coarse* — per una pagina di libro fotografata prova
**prima `sk_model2.pth`**, perché il dettaglio fine è precisamente ciò che stiamo
cercando di eliminare.

*Confidenza: alta sui nomi file e sulla compatibilità dell'architettura; media su
quale checkpoint di `informative-drawings` corrisponda a quale file.* Non
assumere: carica entrambi, misurane F1 con la metrica del § 1, e scegli su base
empirica. Se `load_state_dict` fallisce, stampa le chiavi mancanti/inattese
invece di ripiegare silenziosamente.

Cosa implementare:

1. `scripts/fetch_weights.py` che scarica i pesi via `huggingface_hub`
   (`hf_hub_download(repo_id="lllyasviel/Annotators", filename="sk_model2.pth")`)
   verificando lo SHA, e li deposita nel path atteso dall'engine. Aggiungi
   `huggingface_hub` all'extra `ml`.
2. Riscrivi il messaggio d'errore dei pesi mancanti: deve indicare
   `python scripts/fetch_weights.py`, non un link a Google Drive.
3. Verifica e documenta in `THIRD_PARTY_LICENSES.md` la licenza dei pesi
   ridistribuiti da `lllyasviel/Annotators`. `informative-drawings` è MIT
   (Copyright 2022 Caroline Chan), ma la licenza del *mirror* va accertata
   separatamente. Se non è chiaro, tienilo documentato come tale — la skill
   `add-conversion-style` lo richiede.

---

## 4. Il vero punto debole: la binarizzazione della mappa soft

`anime2sketch` ottiene F1 0.378 — sotto `cartoon` — pur essendo l'unico engine
che *seleziona* correttamente (ignora texture di acqua e roccia, tiene le figure).
Guarda `docs/anime2sketch/starting-image_coloring.jpeg`: le regioni scure
collassano in macchie nere piene (fogliame in basso a sinistra, i sentieri). La
rete fa il lavoro giusto e la binarizzazione lo distrugge.

**Questo è lo stadio che vale più punti di F1, e vale per qualunque engine ML.**
Estrai la binarizzazione in un modulo condiviso, `postprocess.py`, e implementala
così invece che con Otsu globale:

1. **Normalizzazione percentile** della mappa soft (2°/98° percentile), già
   presente.
2. **Estrazione della linea centrale, non soglia d'area.** Una macchia scura
   larga 10 px deve diventare *una* linea, non un blocco pieno. Due vie da
   implementare e confrontare:
   - **NMS lungo la direzione del gradiente** della mappa soft (lo stesso
     principio dello stadio 2 di Canny) prima di sogliare: sopprime tutto ciò che
     non è massimo locale trasversalmente alla linea.
   - **Soglia a isteresi** (due soglie + connettività) seguita da
     `cv2.ximgproc.thinning`, con **pruning dei rami** più corti di N px, perché
     lo scheletro produce rami spuri sulle biforcazioni.
3. **Chain + redraw** riusando lo stadio 3 di `chained`: `approxPolyDP` +
   `polylines` a spessore costante. Non ridisegnare la maschera, ridisegna la
   geometria.

Verifica intermedia attesa: `anime2sketch` da 0.378 a > 0.55 senza toccare la
rete. Se non succede, la diagnosi è sbagliata e va rivista prima di procedere.

---

## 5. La combinazione che ha senso: gating semantico

Qui sta la risposta a "combinare modelli". Non unire engine classici tra loro:
unisci la **geometria pulita** del classico con la **selezione semantica** del
modello ML.

Implementa un engine `--style gated` che:

1. Calcola le catene di bordo con `EdgeDrawing` sull'immagine appiattita
   (stadio 1–2 di `chained`), tarato in modo **permissivo** — vogliamo alta
   recall, la precision la recupera il gate. Parti da `GradientThresholdValue=36,
   MinPathLength=30`.
2. Calcola la mappa soft di `informative_drawings` sulla stessa immagine.
3. Per ogni catena, calcola la **risposta media del modello ML lungo la catena**
   (campiona la mappa soft sui punti del segmento, con un piccolo raggio di
   tolleranza per l'errore di allineamento).
4. **Scarta le catene sotto soglia**, poi ridisegna solo le superstiti con
   `polylines` a spessore uniforme.

Razionale, che segue direttamente dalla decomposizione del § 2: il classico ha
già recall 0.71, quindi le linee giuste ci sono; quello che manca è un criterio
per buttare le altre. Il gradiente non può fornirlo, la mappa ML sì. La soglia del
gate diventa l'unico parametro che scambia precision per recall, ed è tarabile
sulla metrica.

*Questa è un'ipotesi non verificata* — non ho potuto testarla perché mi mancano i
pesi. Misurala appena l'engine del § 3 gira. Se il gating non porta precision
sopra 0.70 mantenendo recall sopra 0.65, abbandonalo e punta tutto su § 4 + § 6.

---

## 6. Secondo stadio: sketch simplification

Se dopo § 3–5 l'F1 è ancora sotto 0.70, il passo successivo è **Sketch
Simplification** (Simo-Serra et al., SIGGRAPH 2016/2018). Non è un estrattore: è
una rete che prende line art grezza e rumorosa e restituisce tratto pulito e
uniforme. È plausibilmente ciò che dà al riferimento MimiPanda il suo aspetto
"disegnato". Va applicata **in cascata** sull'output del miglior engine
precedente, non in alternativa.

Valuta anche `AniLines` (`zhenglinpan/AniLines-Anime-Lineart-Extractor`), più
recente; verifica licenza e dimensione dei pesi prima di integrarlo.

---

## 7. Bug concreti da correggere subito

- **Gli output sono salvati in JPEG.** `docs/*/​*_coloring.jpeg`. La compressione
  JPEG su line art bianco/nero produce ringing attorno a ogni tratto: gonfia la
  copertura di inchiostro misurata e aggiunge rumore reale all'immagine. Il
  formato di default per l'output deve essere **PNG**. In batch mode non ereditare
  l'estensione dell'input.
- **Lo spessore di default è troppo alto.** `chained` ha 16.99 % di inchiostro
  contro 4.37 % del riferimento. Lo sweep mostra che a `thickness=1` la copertura
  scende a 5.09 % con F1 quasi identico (0.574 vs 0.587) — cioè lo spessore
  attuale **gonfia l'inchiostro di 3× senza guadagno di qualità**. Rivedi la
  formula `working_size / 500`: a 1400 px dà ~3 px, troppo. Punta a ~0.15 % del
  lato lungo (1–2 px a 864–1400 px).
- **`adaptive` va rimosso dai choice di default.** F1 0.324, appena sopra il
  baseline degenere di 0.267, con 47 % di inchiostro.

---

## 8. Obiettivo realistico, e perché non è il 95 %

> **Ritirato.** Il criterio "F1 ≥ 0.75 a tolleranza 2 px" qui sotto è stato
> misurato con una metrica che si è poi rivelata rotta: a tolleranza fissa
> 2 px il riferimento stesso, traslato di 3 px, perde contro `canny`
> (0.502 vs 0.614), e un riferimento dilatato fino all'11% di inchiostro
> ottiene F1 = 1.000. Vedi `docs/DIAGNOSIS.md` §0 e `docs/IMPROVEMENT-PROMPT-3.md`
> per la diagnosi completa e la metrica corretta (tolleranza relativa,
> thinning, floor degenere normalizzato, banda di ammissibilità
> sull'inchiostro). I numeri di questa sezione restano come riferimento
> storico di come si è arrivati alla decisione sbagliata su `nms`.

Il target "95 % di accuratezza rispetto a MimiPanda" non è raggiungibile e non è
un obiettivo utile, per una ragione misurabile: **sul benchmark BSDS500 due
annotatori umani che tracciano i bordi della stessa immagine concordano a
F-measure 0.803.** Il riferimento MimiPanda non è una verità oggettiva, è
*un'interpretazione stilistica* — quali dettagli tenere, quanto semplificare,
dove chiudere un contorno. Chiedere 0.95 di accordo con essa è chiedere più
dell'accordo fra due umani sullo stesso compito.

Scala di riferimento, a tolleranza 2 px:

| F1 | significato |
|---:|---|
| 0.27 | baseline degenere (pagina nera) |
| 0.59 | **stato attuale, e tetto dell'approccio classico puro** |
| 0.70 | obiettivo intermedio: § 3 + § 4 |
| 0.75 | obiettivo del progetto: § 5 o § 6 funzionanti |
| 0.80 | accordo fra annotatori umani — soglia oltre cui la metrica misura stile, non qualità |

**Criterio di accettazione: F1 ≥ 0.75 a tolleranza 2 px sulle coppie
input/riferimento disponibili, con copertura di inchiostro entro 3–7 %, e i
baseline degeneri sotto 0.30 nel test suite.** Oltre 0.80 non inseguire il
numero: valuta a occhio.

Nota anche che ci sono solo 3 riferimenti (`desired*.jpg`) contro 7 immagini di
input. Tarare su 3 immagini è sovradattamento garantito. Se vuoi che l'obiettivo
sia credibile, genera con MimiPanda i riferimenti per tutte e 7 le immagini di
`docs/`, e tieni 2 coppie fuori dal tuning come validation set.

---

## Ordine di esecuzione

§ 1 (metrica + baseline degeneri nei test) → § 7 (bug: PNG, spessore, adaptive) →
§ 3 (pesi) → § 4 (binarizzazione) → misura → § 5 (gating) → misura → § 6 solo se
necessario.

Misura F1 dopo ogni fase e riportalo nel messaggio di commit. Commit separati per
fase, conventional commit, nessun push.
