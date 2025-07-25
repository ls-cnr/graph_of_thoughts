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
            1. Include complete, unaltered text segments for each topic.
            2. Create precise, narrowly-focused topics.
            3. Do not omit introductory background information (e.g., name, role, workplace); treat it as a distinct topic.
            4. For Background/Professional Experience topics, include the person’s name and role in the topic title.
            """

def run_example():
    llm_config = LLMConfig(
        name="llama2:7b",
        temperature=0.1,
        top_p=0.9
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
