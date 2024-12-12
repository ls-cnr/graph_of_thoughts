from typing import Dict, List, Type
from src.got.node import GoTNode
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from src.got.keepbest import GoTKeepBest

class GoTAdapter:
    """
    Classe adapter che gestisce il collegamento tra nodi GoT.
    Facilita il passaggio di Thoughts tra nodi diversi gestendo le conversioni necessarie.
    """

    @staticmethod
    def connect_generator_to_keeper(generator: GoTGenerator, keeper: GoTKeepBest) -> None:
        """
        Collega un GoTGenerator a un GoTKeepBest verificando la compatibilità
        e trasferendo gli output nel formato corretto.

        Args:
            generator: Il nodo GoTGenerator che produce gli output
            keeper: Il nodo GoTKeepBest che seleziona il migliore

        Raises:
            ValueError: Se i tipi di Thought non sono compatibili
        """
        # Verifica che il tipo di output del generator sia compatibile con l'input del keeper
        if generator.output_thoughts != keeper.input_thoughts[0]:
            raise ValueError(
                f"Incompatible Thought types: Generator outputs {generator.output_thoughts.__name__}, "
                f"but Keeper accepts {keeper.input_thoughts[0].__name__}"
            )

        # Verifica che il generator produca output multipli
        if generator.output_cardinality == 1:
            raise ValueError(
                "Generator must have output_cardinality != 1 to be connected to a Keeper"
            )

        # Trasforma gli output del generator nel formato atteso dal keeper
        outputs = generator.outputs
        if not outputs:
            raise ValueError("Generator has no outputs to process")

        # Crea il dizionario di input per il keeper
        keeper_inputs = {
            f"thought_{i}": thought
            for i, thought in enumerate(outputs)
        }

        # Processa gli input con il keeper
        keeper.process(keeper_inputs)
