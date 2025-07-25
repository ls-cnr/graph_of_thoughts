from typing import List, Dict, Type
from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from src.got.thought import SentenceThought



class InterviewThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["text", "source"],
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string"}
            }
        }

    def get_for_template(self, key: str) -> str:
        match key:
            case "text":
                return str(self._values["text"])
            case "source":
                return str(self._values["source"])
            case _:
                raise KeyError(f"Campo '{key}' non valido")


class InterviewAnalyzer(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input": InterviewThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return SentenceThought

    @property
    def output_cardinality(self) -> int:
        return -1

    @property
    def task_instruction(self) -> str:
        return """
        You are a precise analyst. Split the interview text into **distinct meaningful sentences**.
        Each sentence should represent **one single informational unit** from the speaker.

        Return one object per sentence with:
        - 'sentence': the exact sentence from the original text.
        - 'context': the source of the interview.
        - 'title': A short, meaningful label for the sentence content (no more than 6-8 words).

        Original text:
        {input.text}
        """

def run_example():
    llm_config = LLMConfig(
        name="llama2:7b",
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=4096
    )

    sample_text = """
    My name is Maria. I am a professional caregiver and I have been working at 'Angeli Custodi' for the past five years.
    My primary responsibility is to offer social and emotional assistance to our elderly guests, many of whom feel isolated.
    I frequently talk with the guests’ family members, updating them about the emotional and physical well-being of their relatives.
    Often, our guests express frustration or sadness because they miss their previous lifestyle.
    Managing this emotional distress is a key part of my daily duties.
    I write a daily report using a system that logs each guest’s activities and issues.
    I also volunteered in a center for refugee women, where emotional support was equally essential.
    """

    input_thought = InterviewThought("input")
    input_thought.values = {
        "text": sample_text,
        "source": "interview_20240329.txt"
    }

    analyzer = InterviewAnalyzer("sentence_analyzer", llm_config)
    analyzer.process({"input": input_thought})

    if analyzer.has_error:
        print(f"Error: {analyzer.error_message}")
    else:
        for i, sentence in enumerate(analyzer.outputs, 1):
            print(f"\n Sentence {i}:")
            print(f"   - Title:    {sentence.values['title']}")
            print(f"   - Context:  {sentence.values['context']}")
            print(f"   - Sentence: {sentence.values['sentence']}")

if __name__ == "__main__":
        run_example()

