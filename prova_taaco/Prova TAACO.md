# TAACO - Test di coesione testuale

Questo script esegue un'analisi automatica di coesione testuale su un insieme di file .txt usando il modulo TAACO_module. I risultati vengono salvati in un file .csv nella cartella di output specificata.

##  Struttura delle cartelle

Assicurati che le cartelle seguenti esistano o vengano correttamente specificate nello script:

- input/: contiene i file .txt da analizzare.
- output/: contiene il file di output risultato.csv generato dal programma.

##  Configurazione

La configurazione dello script è gestita tramite un dizionario varDict che specifica i parametri dell'analisi:

### Parametri principali

| Parametro             | Descrizione |
|-----------------------|-------------|
| language            | Lingua dei testi (english) |
| lemmatize           | Applica la lemmatizzazione |
| wordsContent        | Considera le parole contenuto |
| wordsArgument       | Considera le parole con ruolo argomentale |
| wordsVerb           | Considera i verbi |
| overlapSentence     | Analizza la coesione tra frasi |
| overlapAdjacent     | Analizza la coesione tra frasi adiacenti |
| outputTagged        | Salva anche la versione con tag linguistici |
| outputDiagnostic    | Salva un file diagnostico |

Tutti gli altri parametri sono disattivati (False) per ridurre il rumore nei risultati.

##  Come usare lo script

1. Inserisci i file .txt nella cartella input.
2. Esegui lo script Python.
3. Controlla la cartella output per il file risultato.csv e gli eventuali file aggiuntivi (tagged, diagnostic, ecc.).

##  Output

- risultato.csv: tabella contenente i valori di coesione per ogni file analizzato.
- (Facoltativi) file diagnostici e versioni annotate linguisticamente, se abilitati nei parametri.

##  Dipendenze

- TAACO_module: modulo esterno richiesto per l'esecuzione.
- Libreria standard os.

##  Note

- Il modulo TAACO_module deve essere disponibile nel path del progetto.


### Esecuzione

```bash
python ProvaTAACO.py