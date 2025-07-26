import os
import csv
import networkx as nx
from typing import List, Dict, Type
from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from TAACO_main import TAACO_module

# ====  TAACO ====
class TAACOReticulator:
    def __init__(self, input_dir: str, output_dir: str, llm_config: LLMConfig):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.llm_config = llm_config

    def _clear_input_dir(self):
        for f in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, f))

    def _write_pair_txt(self, t1, t2, fn):
        p = os.path.join(self.input_dir, fn)
        with open(p, "w", encoding="utf-8") as f:
            f.write(t1.values["standalone_sentence"] + "\n")
            f.write(t2.values["standalone_sentence"])
        return p

    def _run_taaco(self, fn, pid) -> float:
        out_csv = os.path.join(self.output_dir, f"taaco_output_{pid}.csv")
        varDict = {
            "language": "english",
            "lemmatize": True,

            # Word categories
            "wordsAll": False,
            "wordsContent": True,
            "wordsFunction": False,
            "wordsNoun": True,  # Aggiunto
            "wordsPronoun": True,  # Aggiunto
            "wordsArgument": True,
            "wordsVerb": True,
            "wordsAdjective": False,
            "wordsAdverb": False,

            # Overlap measures
            "overlapSentence": True,
            "overlapParagraph": False,
            "overlapAdjacent": True,
            "overlapAdjacent2": True,  # Aggiunto: confronto tra frasi distanti di 2
            "overlapLSA": False,
            "overlapLDA": False,
            "overlapWord2vec": False,
            "overlapSynonym": True,  # Aggiunto: coesione basata su sinonimia (senza W2V)

            # Connectives and givenness
            "otherConnectives": True,  # Aggiunto: include coesione tramite connettivi (e.g., because, so)
            "otherGivenness": True,  # Aggiunto: coesione tramite ripetizione di nomi introdotti prima

            # Misc
            "otherTTR": False,
            "sourceLSA": False,
            "sourceLDA": False,
            "sourceWord2vec": False,
            "sourceKeyOverlap": False,
            "outputTagged": True,
            "outputDiagnostic": True,
            "measure_content_overlap": True,
            "content_word_overlap": True
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


    def build_reticle(self, thoughts: List[Thought]):
        for i, current in enumerate(thoughts):
            print()
            print(f"[Reticolo] STEP {i + 1} / {len(thoughts)}")

            # Stampa la frase (standalone) della frase corrente
            print(f"Frase analizzata: \"{current.values['standalone_sentence']}\"")

            nodes = list(self.graph.nodes)

            if not nodes:
                self.graph.add_node(f"n{self.node_counter}", thought=current)
                self.node_counter += 1
                print(" → Primo nodo creato")
                continue

            best, best_coh = None, -1.0

            for j, nid in enumerate(nodes):
                existing = self.graph.nodes[nid]["thought"]

                # Stampa la frase del nodo esistente con cui si confronta
                print(f"\n → Confronto con nodo {nid} ({existing.values['title']}):")
                print(f"     \"{existing.values['standalone_sentence']}\"")

                self._clear_input_dir()
                self._write_pair_txt(existing, current, f"pair{i}_{j}.txt")
                coh = self._run_taaco(None, f"{i}{j}")
                print(f"   → Coesione = {coh:.4f}")

                if coh > best_coh:
                    best_coh, best = coh, existing

            print()
            print(f"   Migliore coesione trovata con nodo '{best.values['title']}':")
            print(f"        \"{best.values['standalone_sentence']}\"")
            print(f"   Valore coesione: {best_coh:.4f}")

            if best_coh > 0.5:
                print(" → Fusione dei pensieri (stessa unità concettuale)")
                bm_node = next(n for n, d in self.graph.nodes(data=True) if d["thought"] == best)
                merged = best.values["standalone_sentence"] + ". " + current.values["standalone_sentence"]
                best.values["standalone_sentence"] = merged
            else:
                new_id = f"n{self.node_counter}"
                self.graph.add_node(new_id, thought=current)
                self.node_counter += 1

                relation_type = "estensione" if best_coh > 0.4 else "scollegato"
                print(f" → Nuovo spazio creato ({relation_type})")

                if relation_type == "estensione":
                    src = next(n for n, d in self.graph.nodes(data=True) if d["thought"] == best)
                    self.graph.add_edge(src, new_id, relation="extension")

        print()
        print("=== Struttura del Reticolo Mentale ===")
        self.print_graph_structure()

    def print_graph_structure(self):
        print("\n=== Mental Space Lattice ===")
        for nid, data in self.graph.nodes(data=True):
            t = data["thought"]
            print(f"{nid}: title='{t.values.get('title')}' | text='{t.values['standalone_sentence']}'")
        for u,v,d in self.graph.edges(data=True):
            print(f"{u} -[{d['relation']}]-> {v}")

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
