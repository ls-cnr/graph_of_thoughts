from typing import List, Dict, Type

from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator


class TextInputThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["text", "max_words"],
            "properties": {
                "text": {"type": "string"},
                "max_words": {"type": "integer"}
            }
        }
    def get_for_template(self, key: str) -> str:
        match key:
            case "text":
                return str(self._values["text"])
            case "max_words":
                return str(self._values["max_words"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per TextThought")

class SummaryThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string"}
            }
        }
    def get_for_template(self, key: str) -> str:
        match key:
            case "summary":
                return str(self._values["summary"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per TextThought")

class TextSummaryGenerator(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input" : TextInputThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return SummaryThought

    @property
    def output_cardinality(self) -> int:
        return 1

    @property
    def task_instruction(self) -> str:
        return """
        Summarize the following text in {input.max_words} words or less.
        Focus on the main points while maintaining clarity and coherence.

        Text: {input.text}
        """

def run_example():
    # Configurazione LLM
    llm_config = LLMConfig(
        name="llama2",
        temperature=0.7,
        top_p=0.9
    )

    # Testo di esempio
    sample_text = """
    Graph of Thoughts (GoT) is a framework that extends the capabilities of Large Language Models
    beyond traditional paradigms like Chain-of-Thought or Tree of Thoughts. It represents units of
    information as vertices (nodes) in a graph, with edges corresponding to dependencies between
    these thoughts. This graph model allows for flexible combination of LLM thoughts, creating
    complex and synergistic structures. Key advantages include flexible combination of thoughts,
    feedback loops for improvement, problem decomposition through divide-and-conquer approaches,
    and reuse of thoughts across different parts of the graph.
    """

    # Crea input thought
    input_thought = TextInputThought("input1")
    input_thought.values = {
        "text": sample_text,
        "max_words": 50
    }

    # Genera il riassunto
    generator = TextSummaryGenerator("summarizer", llm_config)
    generator.process({"input": input_thought})

    # Verifica risultato
    if generator.has_error:
        print(f"Error: {generator.error_message}")
    else:
        for thought in generator.outputs:
            print(f"\nSummary:\n{thought.values['summary']}")

if __name__ == "__main__":
    run_example()
