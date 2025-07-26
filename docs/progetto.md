# Reticolatore Semantico con TAACO 

Questo modulo esegue una **pipeline completa di 
analisi semantica**, decomposizione, ricontestualizzazione 
e viene inoltre costruito un **reticolo di spazi mentali** 
usando metriche di coesione calcolate tramite TAACO.

---

##  Componenti Principali

### 1. **Decontestualizzazione**
- Da un'intervista (testo in input) si estraggono le **frasi e i rispettivi topic principali** con `InterviewAnalyzer`.
- Le frasi vengono **decontestualizzate** con `Decontextualizer`.

### 2. **TAACO Reticulator**
- Costruisce un grafo orientato di pensieri (`Thought`) basato sulla **coesione semantica tra frasi**.
- La coesione è calcolata usando [TAACO](https://lsa.colorado.edu/taaco.html), con parametri personalizzati.
- Le frasi con alta coesione vengono **fuse** nello stesso nodo; le frasi con media coesione vengono estese; invece le frasi con bassa coesione  creano un nuovo nodo nel grafo.
- Ogni nodo ottiene un titolo sintetico.



---

##  Esecuzione della Pipeline

###  Funzione principale: `process_interview()`

1. Legge un'intervista testuale.
2. Analizza il testo in frasi e topic (`InterviewThought → SentenceThought`).
4. Decontestualizza le frasi.
5. Usa `TAACOReticulator` per costruire un **reticolo di frasi coerenti** e genera un documento riassuntivo (`lattice_document.txt`).

---

##  Componente `taaco_reticulator.py`

### `TAACOReticulator`
La classe `TAACOReticulator` costruisce un **reticolo semantico (grafo orientato)** a partire da una lista di pensieri o frasi (`Thought`), utilizzando metriche di coesione testuale fornite da [TAACO](https://lsa.colorado.edu/taaco.html).  
I nodi del grafo rappresentano gruppi di frasi semanticamente affini, mentre gli archi rappresentano relazioni concettuali tra questi gruppi.

---

## Costruttore

### `__init__(input_dir, output_dir, llm_config)`
Inizializza l’istanza del reticolatore TAACO. Prepara le cartelle per input/output dei file usati da TAACO, istanzia il grafo orientato e memorizza la configurazione per il generatore LLM.

```python
def __init__(self, input_dir: str, output_dir: str, llm_config: LLMConfig):
    self.input_dir = input_dir
    self.output_dir = output_dir
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    self.graph = nx.DiGraph()
    self.node_counter = 0
    self.llm_config = llm_config
```

#### Caratteristiche  
- `input_dir` (str): cartella di input per i file `.txt` da analizzare.
- `output_dir` (str): cartella dove salvare i risultati `.csv` di TAACO.
- `llm_config` (`LLMConfig`): configurazione per il generatore di titoli.

#### Scopo  
Preparare l’ambiente di lavoro per costruire dinamicamente un grafo semantico.

---

## Funzioni private

### `_clear_input_dir()`
Cancella tutti i file presenti nella cartella di input per evitare interferenze tra esecuzioni di TAACO.

```python
def _clear_input_dir(self):
    for f in os.listdir(self.input_dir):
        os.remove(os.path.join(self.input_dir, f))
```

#### Caratteristiche  
- Non accetta parametri.
- Elimina ogni file in `self.input_dir`.

#### Scopo  
Assicurarsi che ogni analisi di TAACO sia effettuata su file freschi e univoci.

---

### `_write_pair_txt(t1, t2, fn)` 
Scrive due frasi (`Thought`) su un file `.txt`, ognuna su una riga, pronte per essere analizzate da TAACO.

```python
def _write_pair_txt(self, t1, t2, fn):
    p = os.path.join(self.input_dir, fn)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t1.values["standalone_sentence"] + "\n")
        f.write(t2.values["standalone_sentence"])
    return p
```

#### Caratteristiche  
- `t1`, `t2`: oggetti `Thought`.
- `fn` (str): percorso del file `.txt` da scrivere.

#### Scopo  
Generare file di input leggibili da TAACO per il calcolo della coesione.

---

### `_run_taaco(fn, pid)`ù
Esegue l’analisi di TAACO sul file `.txt` specificato, estrae il file `.csv` prodotto e calcola uno **score aggregato di coesione** pesato su più metriche.

```python
def _run_taaco(self, fn, pid) -> float:
    out_csv = os.path.join(self.output_dir, f"taaco_output_{pid}.csv")
    varDict = {
    "language": "english",  # Imposta la lingua del testo su inglese (obbligatorio)

    "lemmatize": True,  # Applica la lemmatizzazione (riduce le parole alla forma base)

    # Word categories
    "wordsAll": False,  # Non considera tutte le parole
    "wordsContent": True,  # Considera solo le parole contenutistiche (nomi, verbi, ecc.)
    "wordsFunction": False,  # Esclude parole funzionali (preposizioni, articoli, congiunzioni)
    "wordsNoun": True,  # Considera i sostantivi
    "wordsPronoun": True,  # Considera i pronomi
    "wordsArgument": True,  # Considera soggetti e oggetti (argomenti sintattici)
    "wordsVerb": True,  # Considera i verbi
    "wordsAdjective": False,  # Esclude gli aggettivi
    "wordsAdverb": False,  # Esclude gli avverbi

    # Overlap measures
    "overlapSentence": True,  # Calcola l'overlap lessicale tra frasi generiche
    "overlapParagraph": False,  # Disattiva l'analisi tra paragrafi
    "overlapAdjacent": True,  # Analizza coesione tra frasi adiacenti
    "overlapAdjacent2": True,  # Analizza coesione tra frasi separate da una (i.e., distanza 2)
    "overlapLSA": False,  # Disattiva LSA (Latent Semantic Analysis)
    "overlapLDA": False,  # Disattiva LDA (Latent Dirichlet Allocation)
    "overlapWord2vec": False,  # Disattiva Word2Vec per il calcolo della coesione
    "overlapSynonym": True,  # Attiva coesione basata su sinonimia (senza vettori semantici)

    # Connectives and givenness
    "otherConnectives": True,  # Considera la presenza di connettivi logici (e.g., because, so)
    "otherGivenness": True,  # Misura la ripetizione di elementi già menzionati (given-new)

    # Misc
    "otherTTR": False,  # Disattiva il calcolo del Type-Token Ratio (varietà del lessico)
    "sourceLSA": False,  # Non calcola LSA tra frasi sorgente e target
    "sourceLDA": False,  # Non calcola LDA tra frasi sorgente e target
    "sourceWord2vec": False,  # Non usa Word2Vec tra frasi sorgente e target
    "sourceKeyOverlap": False,  # Non considera overlap su parole chiave
    "outputTagged": True,  # Include output con parole POS taggate
    "outputDiagnostic": True,  # Include file di diagnostica per debugging
    "measure_content_overlap": True,  # Calcola esplicitamente overlap tra parole contenutistiche
    "content_word_overlap": True  # Riporta le misure di overlap delle parole contenutistiche
}


    TAACO_module.main(self.input_dir, out_csv, varDict)
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = (
                        0.4 * float(row.get("adjacent_overlap_cw_sent", 0.0)) +
                        0.2 * float(row.get("adjacent_overlap_verb_sent", 0.0)) +
                        0.2 * float(row.get("adjacent_overlap_argument_sent", 0.0)) +
                        0.1 * float(row.get("connectives_sent", 0.0)) +
                        0.1 * float(row.get("givenness_overlap_sent", 0.0))
                )
                return score
            except ValueError:
                return 0.0
    return 0.0
```

#### Motivazione dei pesi

Il punteggio finale tra due frasi è una media pesata di 5 metriche chiave:

```python
score = (
    0.4 * adjacent_overlap_cw_sent +
    0.2 * adjacent_overlap_verb_sent +
    0.2 * adjacent_overlap_argument_sent +
    0.1 * connectives_sent +
    0.1 * givenness_overlap_sent
)
```

#### Caratteristiche  
- `fn` (str): nome del file `.txt` da analizzare.
- `pid`: identificatore nodo (non utilizzato direttamente).
- Restituisce `cohesion_score` (float), che è costituito da:
  - adjacent_overlap_cw_sent (0.4): misura l'overlap delle parole contenutistiche tra frasi vicine. È il principale indicatore di coesione semantica.
  - adjacent_overlap_verb_sent (0.2): i verbi rappresentano azioni e relazioni tra elementi.
  - adjacent_overlap_argument_sent (0.2): individua soggetti, oggetti e attanti del discorso.
  - connectives_sent (0.1): individua relazioni logiche (causa-effetto) tra frasi.
  - givenness_overlap_sent (0.1): rileva elementi già introdotti e ripresi nella frase successiva.

Nota: Questi pesi sono stati scelti empiricamente per favorire l'unione tra frasi semanticamente forti, senza penalizzare le variazioni lessicali o la riformulazione narrativa.

#### Scopo  
Misurare la coesione tra due pensieri per determinare se devono appartenere allo stesso nodo del grafo.

---

## Funzioni pubbliche

### `build_reticle(thoughts: List[Thought])` 
Costruisce il grafo semantico iterando sui pensieri forniti. Ogni pensiero è confrontato con i nodi esistenti e assegnato al nodo più affine, oppure ne genera uno nuovo.

```python
def build_reticle(self, thoughts: List[Thought]):
    for i, current in enumerate(thoughts):
        print(f"[Reticolo] Step {i+1}/{len(thoughts)}")
        nodes = list(self.graph.nodes)
        if not nodes:
            self.graph.add_node(f"n{self.node_counter}", thought=current)
            self.node_counter += 1
            print(" primo nodo creato")
            continue

        best, best_coh = None, -1.0
        for j, nid in enumerate(nodes):
            existing = self.graph.nodes[nid]["thought"]
            self._clear_input_dir()
            self.write_pair_txt(existing, current, f"pair{i}{j}.txt")
            coh = self._run_taaco(None, f"{i}{j}")
            print(f" ↪ confronto con nodo {nid}, coesione={coh:.4f}")
            if coh > best_coh:
                best_coh, best = coh, existing

        print(f" migliore coesione con '{best.values['standalone_sentence']}' = {best_coh:.4f}")
        if best_coh > 0.5:
            print("  → FUSIONE")
            bm_node = next(n for n,d in self.graph.nodes(data=True) if d["thought"]==best)
            merged = best.values["standalone_sentence"] + ". " + current.values["standalone_sentence"]
            best.values["standalone_sentence"] = merged
        else:
            new_id = f"n{self.node_counter}"
            self.graph.add_node(new_id, thought=current)
            self.node_counter +=1
            rel = "extension" if best_coh > 0.2 else "creation"
            print(f"  Nuovo spazio ({'estensione' if rel=='extension' else 'scollegato'})")
            if rel == "extension":
                src = next(n for n,d in self.graph.nodes(data=True) if d["thought"]==best)
                self.graph.add_edge(src, new_id, relation=rel)

    self.print_graph_structure()
```

#### Caratteristiche  
- `thoughts`: lista di oggetti `Thought`.
- Utilizza soglie di coesione.
  - `> 0.5`: unione al nodo esistente.
  - `> 0.3`: nuovo nodo connesso.
  - `≤ 0.3`: nuovo nodo isolato.

#### Motivazione delle soglie di coesione
Le soglie utilizzate per determinare fusione, estensione o creazione di un nuovo nodo nel grafo semantico sono state definite empiricamente, sulla base di osservazioni qualitative durante la fase di sviluppo e test. In particolare:
- `> 0.5`: unione al nodo esistente.
- `> 0.3`: nuovo nodo connesso.
- `≤ 0.3`: nuovo nodo isolato.


#### Scopo  
Organizzare i pensieri in un grafo semantico coerente, che mostra relazioni logiche tra concetti.

---

### `print_graph_structure()` 
Stampa nel terminale la struttura del grafo con titoli e connessioni tra i nodi.

```python
def print_graph_structure(self):
    print("\n=== Mental Space Lattice ===")
    for nid, data in self.graph.nodes(data=True):
        t = data["thought"]
        print(f"{nid}: title='{t.values.get('title')}' | text='{t.values['standalone_sentence']}'")
    for u,v,d in self.graph.edges(data=True):
        print(f"{u} -[{d['relation']}]-> {v}")
```

#### Caratteristiche  
- Non accetta parametri.
- Stampa:
  - Titoli dei nodi.
  - Archi con relazioni (`extension`).

#### Scopo  
Visualizzare la topologia del grafo per verifica o debug.

---

### `generate_lattice_document(output_path)`
Esporta il grafo semantico in un documento testuale. Ogni nodo è rappresentato da un titolo e una lista puntata delle frasi contenute.

```python 
def generate_lattice_document(self, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("DOCUMENTO FINALE: Mental Space Lattice\n")
        f.write("=" * 50 + "\n\n")

        for node_id in self.graph.nodes:
            thought = self.graph.nodes[node_id]["thought"]
            title = thought.values.get("title", "Spazio mentale senza titolo").strip()
            f.write(f"## {title}\n\n")
            sentence = thought.values.get("standalone_sentence", "").strip()
            if sentence:
                f.write("Frasi associate:\n")
                for s in sentence.split(". "):
                    clean_s = s.strip()
                    if clean_s:
                        f.write(f"- {clean_s}\n")
            source = thought.values.get("source", None)
            if source:
                f.write(f"\n*Fonte:* {source}\n")
            f.write("\n" + "-" * 50 + "\n\n")

    print(f"\nDocumento generato in: {output_path}")
```

#### Caratteristiche  
- `output_path` (str): percorso del file `.txt` da scrivere.
- Output in formato leggibile:
  ```text
  ### Titolo del nodo
  - frase 1
  - frase 2
  ```
  
---

# Reticolo Semantico con TAACO 

##  Componente pipiline_project.py
Questa componente definisce una pipeline di elaborazione per trasformare un'intervista in un grafo semantico (reticolo) usando analisi LLM e coesione testuale (TAACO).  
È composto da tre fasi principali:

1. Estrazione di argomenti da un'intervista (InterviewAnalyzer)
3. Decontestualizzazione e costruzione del reticolo semantico con TAACOReticulator

---

### process_interview()
Funzione principale che esegue l'intera pipeline di analisi semantica su un’intervista testuale, composta da tre fasi:
- Analisi intervista e generazione dei topic
- Decontestualizzazione
- Costruzione del reticolo semantico (TAACO)

```python 
def process_interview():
  # LLM config
  llm_config = LLMConfig(
    name="llama2:7b",
    temperature=0.1,
    repeat_penalty=1.2,
    top_p=0.9,
    num_ctx=4096
  )

  # Full input text (context for all sentences)
  sample_text = """
    My name is Maria. I am a professional caregiver and I have been working at 'Angeli Custodi' for the past five years.
    My primary responsibility is to offer social and emotional assistance to our elderly guests, many of whom feel isolated.
    I frequently talk with the guests’ family members, updating them about the emotional and physical well-being of their relatives.
    Often, our guests express frustration or sadness because they miss their previous lifestyle.
    Managing this emotional distress is a key part of my daily duties.
    I write a daily report using a system that logs each guest’s activities and issues.
    I also volunteered in a center for refugee women, where emotional support was equally essential.
    """

  # Step 1: Analisi dell'intervista → frasi
  interview = InterviewThought("initial_interview")
  interview.values = {
    "text": sample_text,
    "source": "interview_20240329.txt"
  }

  analyzer = InterviewAnalyzer("sentence_splitter", llm_config)
  analyzer.process({"input": interview})

  if analyzer.has_error:
    print(f"Analyzer error: {analyzer.error_message}")
    return

  sentence_thoughts = analyzer.outputs

  # Step 2: Decontestualizza ogni frase

  # MODIFICA: crea il decontestualizzatore UNA SOLA VOLTA
  decontextualizer = Decontextualizer("shared_decontext", llm_config)

  decontextualized_thoughts = []
  for i, sent in enumerate(sentence_thoughts):

    # Costruiamo un nuovo SentenceThought con contesto completo
    enriched_sentence = SentenceThought(f"sent_{i}")
    enriched_sentence.values = {
      "sentence": sent.values["sentence"],
      "context": sample_text.strip(),  # contesto completo
      "title": sent.values.get("title", ""),
      "source": sent.values.get("context", "")
    }

    try:
      # MODIFICA: riutilizziamo la stessa istanza
      decontextualizer.process({"input": enriched_sentence})

      if decontextualizer.has_error:
        print(f"[ERROR] Decontext error at sentence {i}: {decontextualizer.error_message}")
        continue

      decontextualized_thoughts.extend(decontextualizer.outputs)
      print(f"[DEBUG] Done with sentence {i + 1}")

    except Exception as e:
      print(f"[EXCEPTION] Failed at sentence {i}: {e}")

  # Step 3: Output dei risultati
  print("\n=== Decontextualized Results ===")
  for i, thought in enumerate(decontextualized_thoughts, 1):
    print(f"\n{i}. Original:   {thought.values['in_context_sentence']}")
    print(f"   Standalone: {thought.values['standalone_sentence']}")
    print(f"   Title:      {thought.values.get('title', '')}")

    # Step 4: Lattice
    import os
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TAACO_main')))
    from examples.merge.taaco_reticulator import TAACOReticulator

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.join(base_dir, "taaco_input", "input")
    output_dir = os.path.join(base_dir, "taaco_output")
    lattice_output_file = os.path.join(base_dir, "output", "lattice_document.txt")

    reticulator = TAACOReticulator(input_dir, output_dir, llm_config)
    reticulator.build_reticle(decontextualized_thoughts)
    reticulator.generate_lattice_document(lattice_output_file)
```

**Caratteristiche**
- L'intervista viene elaborata per estrarre SentenceThought, ovvero concetti principali trattati nel testo.
- Successivamente, la frase viene decontestualizzata: trasformata in una frase autonoma e comprensibile senza riferimento esplicito al testo originale.
- Utilizza la componente TAACO per calcolare la coesione semantica tra frasi.
- Genera un grafo orientato (reticolo) dove:
  - Le frasi simili vengono fuse. 
  - Le frasi parzialmente coerenti si collegano come estensioni. 
  - Le frasi poco coerenti creano nuovi nodi indipendenti.
- Produce infine un documento .txt che rappresenta il reticolo semantico finale.

---

##  Output finale

Il sistema genera un file lattice_document.txt che contiene:

- Titoli generati per ciascuno spazio mentale.
- Frasi decontestualizzate associate.

---

##  Requisiti

- Python ≥ 3.8
- LLM compatibile con GoTGenerator e LLMConfig
- TAACO
- Librerie: NetworkX, nltk, spacy, torch

---

## Installazione e uso

Abbiamo utilizzato TAACO nella versione Python (CLI) anziché come eseguibile `.jar`, perché:

- è più facilmente integrabile nel codice
- evita la gestione di processi esterni o installazioni Java
- consente un controllo diretto sui parametri
- permette il debug del contenuto generato dinamicamente

TAACO è stato scaricato da:
https://github.com/nlsdfg/TAACO ????

---

## Note
- È pensato per analisi semantiche complesse, utile in studi qualitativi, interviste, analisi narrative.

---

##  Avvio rapido

```bash
# Assicurati di essere nella cartella giusta
cd graph_of_thoughts

# Esegui lo script
python -m examples.decontextualization.interview_decontext
```
