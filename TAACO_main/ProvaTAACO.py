import os
import TAACO_module

# Percorso della cartella di input (dove si trovano i file .txt)
input_dir = "C:/Users/HP/Desktop/input_taaco"

# Percorso della cartella di output (dove salvare il file CSV)
output_folder = "C:/Users/HP/Desktop/output_taaco"
output_file = os.path.join(output_folder, "risultato.csv")

# Assicurati che la cartella di output esista
os.makedirs(output_folder, exist_ok=True)

# Dizionario di configurazione
varDict = {
    "language": "english",
    "lemmatize": False,

    "wordsAll": False,
    "wordsContent": True,
    "wordsFunction": False,
    "wordsNoun": False,
    "wordsPronoun": False,
    "wordsArgument": True,
    "wordsVerb": True,
    "wordsAdjective": False,
    "wordsAdverb": False,

    "overlapSentence": True,
    "overlapParagraph": False,
    "overlapAdjacent": True,
    "overlapAdjacent2": False,
    "otherConnectives": False,
    "otherGivenness": False,

    "otherTTR": False,
    "overlapLSA": False,
    "overlapLDA": False,
    "overlapWord2vec": False,
    "overlapSynonym": False,

    "sourceLSA": False,
    "sourceLDA": False,
    "sourceWord2vec": False,

    "outputTagged": True,
    "outputDiagnostic": True,

    "sourceKeyOverlap": False
}


print("Avvio TAACO...")

# Esegui TAACO
TAACO_module.main(input_dir, output_file, varDict)

# Mostra il contenuto della cartella di output
print("Contenuto della cartella di output:")
print(os.listdir(output_folder))
