from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod

from src.got.node import GoTNode, LLMConfig
from src.got.generator import GoTGenerator
from src.got.thought import Thought

class GoTRepeat(GoTNode):
    """
    Nodo che esegue K volte un generator embedded per produrre output multipli.
    Utile per generare diverse varianti partendo dallo stesso input.
    """

    def __init__(self, node_id: str, llm_config: LLMConfig, embedded_generator: GoTGenerator, k: int):
        """
        Inizializza un nodo GoTRepeat.

        Args:
            node_id: Identificatore univoco del nodo
            llm_config: Configurazione del modello LLM
            embedded_generator: Istanza del generator da ripetere
            k: Numero di ripetizioni da eseguire
        """
        super().__init__(node_id, llm_config)
        self.embedded_generator = embedded_generator
        self.k = k

        # Valida k
        if k <= 0:
            raise ValueError("k must be positive")

    @property
    def input_thoughts(self) -> List[Type[Thought]]:
        """Lista dei tipi di Thought accettati come input (delegato al generator embedded)"""
        return self.embedded_generator.input_thoughts

    @property
    def output_thoughts(self) -> Type[Thought]:
        """Tipo di Thought prodotto come output (delegato al generator embedded)"""
        return self.embedded_generator.output_thoughts

    @property
    def output_cardinality(self) -> int:
        """
        Cardinalità degli output. Per GoTRepeat è sempre k * generator_cardinality.
        Se il generator produce un singolo output, GoTRepeat ne produrrà k.
        Se il generator produce n output, GoTRepeat ne produrrà k*n.
        """
        base_cardinality = (
            1 if self.embedded_generator.output_cardinality == 1
            else len(self.embedded_generator.outputs)
        )
        return self.k * base_cardinality

    def process(self, inputs: Dict[str, Thought]) -> None:
        """
        Processa gli input ripetendo k volte il generator embedded.

        Args:
            inputs: Dizionario degli input per il generator
        """
        try:
            all_outputs = []

            for i in range(self.k):
                # Clona il generator per ogni iterazione per evitare interferenze
                iteration_generator = type(self.embedded_generator)(
                    f"{self.node_id}_iter_{i}",
                    self.llm_config
                )

                # Processa gli input con il generator clonato
                iteration_generator.process(inputs)

                if iteration_generator.has_error:
                    raise ValueError(
                        f"Embedded generator failed at iteration {i}: "
                        f"{iteration_generator.error_message}"
                    )

                # Aggiungi gli output di questa iterazione
                all_outputs.extend(iteration_generator.outputs)

            # Assegna tutti gli output generati
            self.outputs = all_outputs

        except Exception as e:
            error_msg = f"Error in {self.node_id}: {str(e)}"
            self.set_error(error_msg)
