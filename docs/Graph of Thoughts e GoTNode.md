# Graph of Thoughts (GoT)

Graph of Thoughts è un framework che estende le capacità di prompting dei Large Language Models (LLMs) oltre i paradigmi tradizionali come Chain-of-Thought o Tree of Thoughts.

## Concetti chiave
- **Vertici (Nodi)**: Rappresentano unità di informazione ("LLM thoughts") generate dal modello linguistico
- **Archi**: Corrispondono alle dipendenze tra i vertici - indicano come le informazioni fluiscono da un pensiero all'altro
- **Modello a grafo**: Permette di combinare arbitrariamente i pensieri dell'LLM, creando strutture complesse e sinergiche

## Vantaggi
1. **Combinazione flessibile**: Possibilità di unire diversi pensieri per ottenere risultati sinergici
2. **Feedback loop**: Capacità di migliorare i pensieri attraverso cicli di feedback
3. **Decomposizione dei problemi**: Approccio divide-et-impera per problemi complessi
4. **Riutilizzo**: I pensieri possono essere riutilizzati in parti diverse del grafo

# Guida all'Implementazione

## Panoramica
Il Graph of Thoughts è un'architettura che permette di costruire grafi di elaborazione in cui ogni nodo può:
- Generare contenuti usando LLM (GoTGenerator)
- Valutare contenuti assegnando score (GoTKeepBest)
- Ripetere operazioni multiple volte (GoTRepeat)

## Componenti Principali

### Thought - Unità Base di Informazione
La classe `Thought` rappresenta l'unità base di informazione nel grafo:

```python
class Thought:
    def __init__(self, thought_id: str, values: Optional[Dict[str, Any]] = None):
        self.thought_id = thought_id
        self._values = values or {}
```

**Caratteristiche:**
- Schema JSON per validazione dati
- Validazione automatica dei valori
- Formattazione personalizzabile per template

### GoTNode - Classe Base
GoTNode è la classe base astratta che fornisce le funzionalità comuni a tutti i nodi:

```python
class GoTNode(ABC):
    def __init__(self, node_id: str, llm_config: LLMConfig):
        self.node_id = node_id
        self.llm_config = llm_config
        self._outputs: List[Thought] = []
        self._has_error: bool = False
        self._error_message: Optional[str] = None
```

**Metodi Astratti:**
- `input_thoughts`: tipi di Thought accettati come input
- `output_thoughts`: tipo di Thought prodotto come output
- `output_cardinality`: numero di output prodotti
- `process`: elaborazione degli input

**Funzionalità Comuni:**
- Configurazione LLM
- Gestione output e stato di errore
- Meccanismo di retry (MAX_RETRIES = 3)

### GoTGenerator - Generazione di Contenuti
GoTGenerator è specializzato nella generazione di contenuti tramite LLM:

```python
class MyGenerator(GoTGenerator):
    @property
    def task_instruction(self) -> str:
        return "Istruzioni per l'LLM..."

    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input1": ThoughtType1, "input2": ThoughtType2}
```

**Caratteristiche:**
- Definisce il mapping tra nomi logici e tipi di Thought
- Gestisce la generazione del template completo
- Estrae e valida JSON dall'output dell'LLM
- Supporta la generazione di output multipli

### GoTKeepBest - Selezione del Migliore
GoTKeepBest seleziona il migliore tra più Thought dello stesso tipo:

```python
class MyKeeper(GoTKeepBest):
    @property
    def input_thoughts(self) -> List[Type[Thought]]:
        return [MyThoughtType]

    def _assign_scores(self, input: Thought) -> Dict[str, float]:
        return {"score1": 0.8, "score2": 0.6}

    def _compare(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> int:
        return sum(scores1.values()) - sum(scores2.values())
```

**Caratteristiche:**
- Assegna score multipli a ogni input
- Implementa logica di confronto personalizzata
- Produce sempre un singolo output (il migliore)

### GoTRepeat - Ripetizione di Operazioni
GoTRepeat esegue un generator multipli volte per produrre varianti:

```python
class MyRepeater(GoTRepeat):
    def __init__(self, node_id: str, llm_config: LLMConfig, generator: GoTGenerator, k: int):
        super().__init__(node_id, llm_config)
        self.embedded_generator = generator
        self.k = k
```

**Caratteristiche:**
- Ripete l'esecuzione del generator k volte
- Combina tutti gli output in un'unica lista
- Propaga eventuali errori del generator

## Pattern di Output
Ogni tipo di nodo ha un pattern di output specifico:
- **Generator**: Thought singolo o lista in base alla cardinalità
- **KeepBest**: Singolo Thought (il migliore)
- **Repeat**: Lista di k * n Thought (dove n è la cardinalità del generator)

## Gestione Errori
Tutti i nodi implementano:
- Validazione degli input richiesti
- Validazione degli output tramite schema
- Retry automatici in caso di errore
- Tracciamento dello stato di errore
- Messaggi di errore dettagliati

## Best Practices

### 1. Definizione dei Thought
- Definire schemi JSON precisi
- Implementare la formattazione per template
- Validare sempre i dati

### 2. Implementazione dei Generator
- Definire istruzioni chiare per l'LLM
- Specificare correttamente il mapping degli input
- Gestire la cardinalità degli output

### 3. Implementazione dei Keeper
- Definire metriche di scoring significative
- Implementare confronti consistenti
- Gestire casi limite negli score

### 4. Gestione Errori
- Validare tutti gli input
- Fornire messaggi di errore informativi
- Implementare retry quando appropriato

### 5. Documentazione
- Documentare il comportamento atteso
- Specificare i tipi di input/output
- Descrivere la logica di elaborazione
