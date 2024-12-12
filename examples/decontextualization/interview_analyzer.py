from typing import List, Dict, Type

from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator


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
                raise KeyError(f"Campo '{key}' non valido per SentenceThought")


class TopicThought(Thought):
    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["topic_name", "content", "source"],
            "properties": {
                "topic_name": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string"}
            }
        }

    def get_for_template(self, key: str) -> str:
        match key:
            case "topic_name":
                return str(self._values["topic_name"])
            case "content":
                return str(self._values["content"])
            case "source":
                return str(self._values["source"])
            case _:
                raise KeyError(f"Campo '{key}' non valido per SentenceThought")


class InterviewAnalyzer(GoTGenerator):
    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input" : InterviewThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return TopicThought

    @property
    def output_cardinality(self) -> int:
        return -1

    @property
    def task_instruction(self) -> str:
        return """
        You are an expert knowledge analyst. Analyze the following text and extract distinct topics.
        Focus on fine-grained decomposition with specific rather than broad topics.

        Text: {input.text}

        Instructions:
        1. Include complete, unaltered text segments for each topic
        2. Create precise, narrowly-focused topics
        3. For Background/Professional Experience topics, include person's name/role in title
        """

def run_example():
    llm_config = LLMConfig(
        name="llama2",
        temperature=0.1,
        top_p=0.9
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

    input_thought = InterviewThought("input")
    input_thought.values = {
        "text": sample_text,
        "source": "interview_20240329.txt"
    }

    analyzer = InterviewAnalyzer("topic_analyzer", llm_config)
    analyzer.process({"input": input_thought})

    if analyzer.has_error:
        print(f"Error: {analyzer.error_message}")
    else:
        for topic in analyzer.outputs:
            print(f"\nTopic: {topic.values['topic_name']}")
            print(f"Content: {topic.values['content']}")

if __name__ == "__main__":
    run_example()
