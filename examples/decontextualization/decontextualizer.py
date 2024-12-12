from typing import List, Dict, Type

from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator


class SentenceThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["sentence", "context"],
            "properties": {
                "sentence": {"type": "string"},
                "context": {"type": "string"}
            }
        }

    def get_for_template(self, key: str) -> str:
        match key:
            case "sentence":
                return str(self._values["sentence"])
            case "context":
                return str(self._values["context"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per SentenceThought")


class DecontextualizedThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["in_context_sentence", "standalone_sentence"],
            "properties": {
                "in_context_sentence": {"type": "string"},
                "standalone_sentence": {"type": "string"}
            }
        }
        def get_for_template(self, key: str) -> str:
            match key:
                case "in_context_sentence":
                    return str(self._values["in_context_sentence"])
                case "standalone_sentence":
                    return str(self._values["standalone_sentence"])
                case _:
                    raise KeyError(f"Campo '{key}' non valido per SentenceThought")

class Decontextualizer(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input" : SentenceThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return DecontextualizedThought

    @property
    def output_cardinality(self) -> int:
        return 1

    @property
    def task_instruction(self) -> str:
        return """
        Your task is to take the provided sentence and its surrounding context, and rewrite the sentence
        in a way that makes it self-contained and interpretable without the original context.

        Original sentence: {input.sentence}
        Context: {input.context}

        The goal is to preserve the meaning while adding necessary clarifying details:
        - Replace pronouns with relevant names/roles
        - Add contextual information (location, time, profession)
        - Modify minimally, only to decontextualize
        Do not summarize or paraphrase. Maintain the original meaning.
        """

def run_example():
    llm_config = LLMConfig(
        name="mistral:instruct",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9,
        num_ctx=4096
    )

    sample_text = """
    My name is Maria. I am a professional caregiver. I am working in 'Angeli Custodi' since 5 years.
    My main responsibility is to provide social assistance to guests.
    My work is also to receive and talk with guests' relatives.
    We must handle guests' anxiety that is due to their desire to live normally,
    have social relationships and receive a good service.
    I use a computer to write the daily report. Sometimes I use to take notes in a block note during my shift,
    so I must to re-write all before leaving.
    """

    input_thought = SentenceThought("input")
    input_thought.values = {
        "sentence": "I must to re-write all before leaving",
        "context": sample_text
    }

    decontextualizer = Decontextualizer("decontextualizer", llm_config)
    decontextualizer.process({"input": input_thought})

    if decontextualizer.has_error:
        print(f"Error: {decontextualizer.error_message}")
    else:
        result = decontextualizer.outputs[0].values
        print(f"\nOriginal: {result['in_context_sentence']}")
        print(f"Standalone: {result['standalone_sentence']}")

if __name__ == "__main__":
    run_example()
