from typing import List, Dict, Type

from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator

class TextThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"}
            }
        }
    def get_for_template(self, key: str) -> str:
        match key:
            case "text":
                return str(self._values["text"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per TextThought")

class MergedTextThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["merged_text"],
            "properties": {
                "merged_text": {"type": "string"}
            }
        }
    def get_for_template(self, key: str) -> str:
        match key:
            case "text":
                return str(self._values["text"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per MergedTextThought")


class TextMerger(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"text1" : TextThought, "text2" : TextThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return MergedTextThought

    @property
    def output_cardinality(self) -> int:
        return 1

    @property
    def task_instruction(self) -> str:
        return """
        Merge these two texts into a coherent single text:

        First text: {text1.text}

        Second text: {text2.text}
        """

def run_example():
    llm_config = LLMConfig(
        name="llama2",
        temperature=0.7,
        top_p=0.9
    )

    text1 = TextThought("text1")
    text1.values = {
        "text": "Graph of Thoughts represents information as nodes in a graph."
    }

    text2 = TextThought("text2")
    text2.values = {
        "text": "The edges in the graph show how different thoughts are connected."
    }

    merger = TextMerger("merger", llm_config)
    merger.process({"text1": text1, "text2": text2})

    if merger.has_error:
        print(f"Error: {merger.error_message}")
    else:
        print(f"\nMerged text:\n{merger.outputs[0].values['merged_text']}")

if __name__ == "__main__":
    run_example()
