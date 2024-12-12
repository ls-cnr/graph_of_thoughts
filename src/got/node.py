from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from jsonschema import validate, ValidationError
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.got.thought import Thought

@dataclass
class LLMConfig:
    """Configurazione del modello LLM"""
    name: str
    temperature: float = 0.1
    repeat_penalty: float = 1.2
    top_p: float = 0.9
    num_ctx: int = 4096

class GoTNode(ABC):
    """Nodo base astratto per Graph of Thoughts"""

    MAX_RETRIES = 3

    def __init__(self, node_id: str, llm_config: LLMConfig):
        self.node_id = node_id
        self.llm_config = llm_config
        self._outputs: List[Thought] = []
        self._has_error: bool = False
        self._error_message: Optional[str] = None

        if llm_config:
            self.llm = Ollama(
                model=self.llm_config.name,
                temperature=self.llm_config.temperature,
                repeat_penalty=self.llm_config.repeat_penalty,
                top_p=self.llm_config.top_p,
                num_ctx=self.llm_config.num_ctx,
            )

    @property
    @abstractmethod
    def input_thoughts(self) -> List[Type[Thought]]:
        """Lista dei tipi di Thought accettati come input"""
        pass

    @property
    @abstractmethod
    def output_thoughts(self) -> Type[Thought]:
        """Lista dei tipi di Thought prodotti come output"""
        pass

    @property
    @abstractmethod
    def output_cardinality(self) -> int:
        """
        Numero di output che il nodo deve produrre.
        Un valore di -1 indica che sarà restituita una lista.

        Returns:
            int: Numero esatto di output richiesti
        """
        pass

    @abstractmethod
    def process(self, inputs: Dict[str, Thought]) -> None:
        pass

    @property
    def outputs(self) -> List[Thought]:
        """Getter per i Thought di output prodotti"""
        return self._outputs

    @outputs.setter
    def outputs(self, thoughts: List[Thought]) -> None:
        self._outputs = thoughts
        # Reset dello stato di errore quando vengono impostati nuovi output
        self._has_error = False
        self._error_message = None

    @property
    def has_error(self) -> bool:
        """Indica se si è verificato un errore durante il processo"""
        return self._has_error

    @property
    def error_message(self) -> Optional[str]:
        """Messaggio di errore se si è verificato un problema"""
        return self._error_message

    def set_error(self, message: str) -> None:
        """
        Imposta lo stato di errore del nodo.

        Args:
            message: Descrizione dell'errore
        """
        self._has_error = True
        self._error_message = message
        self._outputs = []  # Pulisce gli output in caso di errore
