from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import json
from jsonschema import validate, ValidationError

class Thought:
    """
    Classe base che rappresenta un'unità di informazione (thought) nel grafo.
    Contiene un dizionario di valori e uno schema per la loro validazione.
    """
    def __init__(self, thought_id: str, values: Optional[Dict[str, Any]] = None):
        self.thought_id = thought_id
        self._values = values or {}

    @property
    @abstractmethod
    def schema(self) -> dict:
        """Schema JSON che definisce la struttura attesa dei valori"""
        pass

    @property
    def values(self) -> Dict[str, Any]:
        """I valori contenuti nel thought"""
        return self._values

    @abstractmethod
    def get_for_template(self, key: str) -> str:
        """
        Restituisce una rappresentazione formattata del valore per l'uso nei template.
        Ogni sottoclasse deve implementare la propria logica di formattazione.

        Args:
            key: Chiave del valore da formattare

        Returns:
            str: Rappresentazione formattata del valore

        Raises:
            KeyError: Se la chiave non esiste nei valori del thought
        """
        pass

    @values.setter
    def values(self, new_values: Dict[str, Any]):
        """Aggiorna i valori senza validazione"""
        self._values = new_values

    def is_valid(self) -> bool:
        """
        Verifica se i valori correnti rispettano lo schema.

        Returns:
            bool: True se i valori sono validi, False altrimenti
        """
        try:
            validate(instance=self._values, schema=self.schema)
            return True
        except ValidationError:
            return False

    def validate(self) -> None:
        """
        Valida i valori correnti contro lo schema.

        Raises:
            ValidationError: Se i valori non rispettano lo schema
        """
        validate(instance=self._values, schema=self.schema)


class InterviewThought(Thought):
    """Thought che rappresenta un'intervista da analizzare"""

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["interview_text", "source"],
            "properties": {
                "interview_text": {"type": "string"},
                "source": {"type": "string"}
            },
            "additionalProperties": False
        }


class TopicsThought(Thought):
    """Thought che rappresenta i topics estratti da un'intervista"""

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["topics"],
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["topic_name", "content", "source"],
                        "properties": {
                            "topic_name": {"type": "string"},
                            "content": {"type": "string"},
                            "source": {"type": "string"}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        }


# Esempio di utilizzo
if __name__ == "__main__":
    # Creazione di un InterviewThought
    interview = InterviewThought("interview_1")

    # Impostazione di valori validi
    interview.values = {
        "interview_text": "Example interview text",
        "source": "interview_001.txt"
    }
    print(f"Interview thought valid? {interview.is_valid()}")  # True

    # Impostazione di valori non validi
    interview.values = {
        "wrong_field": "this won't validate"
    }
    print(f"Interview thought valid? {interview.is_valid()}")  # False

    # La validazione esplicita solleverà un'eccezione
    try:
        interview.validate()
    except ValidationError as e:
        print(f"Validation failed: {e}")

    # Creazione di un TopicsThought
    topics = TopicsThought("topics_1")
    topics.values = {
        "topics": [
            {
                "topic_name": "First Topic",
                "content": "Content of first topic",
                "source": "interview_001.txt"
            }
        ]
    }
    print(f"Topics thought valid? {topics.is_valid()}")  # True
