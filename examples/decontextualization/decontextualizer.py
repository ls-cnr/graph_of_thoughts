from typing import List, Dict, Type

from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from src.got.thought import SentenceThought

class DecontextualizedThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["in_context_sentence", "standalone_sentence"],
            "properties": {
                "in_context_sentence": {"type": "string"},
                "standalone_sentence": {"type": "string"},
                "title": {"type": "string"},   # ← Aggiunto
                "source": {"type": "string"}   # ← Aggiunto
            }
        }

    def get_for_template(self, key: str) -> str:
        match key:
            case "in_context_sentence":
                return str(self._values["in_context_sentence"])
            case "standalone_sentence":
                return str(self._values["standalone_sentence"])
            case "title":
                return str(self._values.get("title", ""))
            case "source":
                return str(self._values.get("source", ""))
            case _:
                raise KeyError(f"Campo '{key}' non valido per DecontextualizedThought")


class Decontextualizer(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input": SentenceThought}

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
            so that it is completely self-contained and interpretable without the original context.

            Original sentence: {input.sentence}
            Context: {input.context}

            The goal is to preserve the original meaning while making the sentence standalone.

            STRICT RULES:
            - Use ONLY information found explicitly in the context.
            - DO NOT invent or add information not present in the context.
            - Replace ALL pronouns (I, we, she, he, they, etc.) with the appropriate noun or name,
              IF AND ONLY IF the referent is clearly mentioned in the context.
            - Do NOT change locations, names, or introduce new elements.
            - Do NOT summarize, paraphrase or generalize. Just make the sentence interpretable without needing the rest of the context.

            VERY IMPORTANT:
            If a pronoun appears, and its referent is in the context, you MUST replace it with that referent.
            Otherwise, leave it unchanged.

            Output ONLY the rewritten sentence, nothing else.
           """

    def process(self, inputs: Dict[str, Thought]) -> None:
        super().process(inputs)

        if self.has_error or not self.outputs:
            return

        input_thought = inputs["input"]
        output_thought = self.outputs[0]

        # Propaga i metadati title e source, se presenti
        for key in ["title", "source"]:
            if key in input_thought.values:
                output_thought.values[key] = input_thought.values[key]


def run_example():
    llm_config = LLMConfig(
        name="llama2:7b",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9,
        num_ctx=4096
    )

    sample_text = """
   My name is Maria. I am a professional caregiver and I have been working at 'Angeli Custodi' for the past five years.  
My primary responsibility is to offer social and emotional assistance to our elderly guests, many of whom feel isolated.  
I frequently talk with the guests’ family members, updating them about the emotional and physical well-being of their relatives.  

Often, our guests express frustration or sadness because they miss their previous lifestyle — the ability to go out, meet friends, or live independently.  
Managing this emotional distress is a key part of my daily duties. I try to reassure them and involve them in group activities to improve their mood.  

To document all the relevant information, I write a daily report using a computer system that logs each guest’s activities and issues.  
During my shift, I take quick notes in a notebook, which I later transcribe into the system before leaving.  
Sometimes the software is slow, and this delays my exit, but it’s essential to leave accurate records for the next shift.  

Separately, I have also volunteered in a center for refugee women, where the emotional support needed is quite different, but equally important.  
There, I conducted weekly conversation groups to help participants share their experiences and regain confidence in a new cultural environment.

    """

    input_thought = SentenceThought("input")
    input_thought.values = {
        "sentence": "I must to re-write all before leaving",
        "context": sample_text,
        "title": "Logging Daily Reports",
        "source": "interview_20240329.txt"
    }

    decontextualizer = Decontextualizer("decontextualizer", llm_config)
    decontextualizer.process({"input": input_thought})

    if decontextualizer.has_error:
        print(f"Error: {decontextualizer.error_message}")
    else:
        result = decontextualizer.outputs[0].values
        print(f"\nOriginal: {result['in_context_sentence']}")
        print(f"Standalone: {result['standalone_sentence']}")
        print(f"Title: {result.get('title')}")


if __name__ == "__main__":
    run_example()
