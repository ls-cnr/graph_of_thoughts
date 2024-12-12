from typing import Dict, List, Any, Optional, Type
from abc import abstractmethod

from src.got.node import GoTNode, LLMConfig
from src.got.thought import Thought

class GoTKeepBest(GoTNode):
    """
    Nodo che seleziona il migliore tra più Thought dello stesso tipo
    basandosi su una funzione di scoring personalizzabile.
    """

    def __init__(self, node_id: str, llm_config: LLMConfig):
        """
        Inizializza il nodo GoTKeepBest.

        Args:
            node_id: Identificatore univoco del nodo
            llm_config: Configurazione LLM (opzionale per questo nodo)
        """
        super().__init__(node_id, llm_config)
        self._scored_items: List[Dict[str, Any]] = []

    @property
    @abstractmethod
    def input_thoughts(self) -> List[Type[Thought]]:
        """
        Definisce il tipo di Thought accettato come input.
        Deve restituire una lista con un solo tipo.
        """
        pass

    @property
    def output_thoughts(self) -> Type[Thought]:
        """
        Il tipo di output è lo stesso del tipo di input.
        """
        return self.input_thoughts[0]

    @property
    def output_cardinality(self) -> int:
        """
        Questo nodo produce sempre un singolo output.
        """
        return 1

    @abstractmethod
    def _assign_scores(self, input: Thought) -> Dict[str, float]:
        """
        Assegna punteggi al Thought di input.

        Args:
            input: Il Thought da valutare

        Returns:
            Dizionario con nome_score: valore_score
        """
        pass

    @abstractmethod
    def _compare(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> int:
        """
        Confronta due set di punteggi.

        Args:
            scores1: Primo set di punteggi da confrontare
            scores2: Secondo set di punteggi da confrontare

        Returns:
            int: >0 se scores1 è migliore, <0 se scores2 è migliore, 0 se equivalenti
        """
        pass

    def process(self, inputs: Dict[str, Thought]) -> None:
        """
        Processa gli input calcolando gli score e selezionando il migliore.

        Args:
            inputs: Dizionario di Thought da processare
        """
        try:
            # Calcola gli score per ogni input
            self._scored_items = []
            for thought_id, thought in inputs.items():
                try:
                    scores = self._assign_scores(thought)
                    self._scored_items.append({
                        "thought": thought,
                        "scores": scores
                    })
                except Exception as e:
                    self.set_error(f"Error scoring thought {thought_id}: {str(e)}")
                    return

            if not self._scored_items:
                self.set_error("No valid scored items found")
                return

            # Trova il migliore confrontando gli score
            best_item = self._scored_items[0]
            for item in self._scored_items[1:]:
                comparison = self._compare(item["scores"], best_item["scores"])
                if comparison > 0:
                    best_item = item

            # Imposta l'output come il thought migliore
            self.outputs = [best_item["thought"]]

        except Exception as e:
            error_msg = f"Error in {self.node_id}: {str(e)}"
            self.set_error(error_msg)
