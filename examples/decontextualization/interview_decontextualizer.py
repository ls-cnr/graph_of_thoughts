from typing import Dict, Type, List

from src.got.node import GoTNode,LLMConfig
from src.got.thought import Thought
from .interview_analyzer import InterviewAnalyzer, InterviewThought, TopicThought
from .decontextualizer import Decontextualizer, SentenceThought, DecontextualizedThought



class TopicToSentenceBridge(GoTNode):
    """Bridge to convert TopicThought to SentenceThought for Decontextualizer"""

    def __init__(self, node_id: str, llm_config: LLMConfig):
        super().__init__(node_id, llm_config)

    @property
    def input_thoughts(self) -> List[Type[Thought]]:
        return [TopicThought]

    @property
    def output_thoughts(self) -> Type[Thought]:
        return SentenceThought

    @property
    def output_cardinality(self) -> int:
        return 1

    def process(self, inputs: Dict[str, Thought]) -> None:
        try:
            topic = list(inputs.values())[0]
            sentence = SentenceThought(f"{self.node_id}_output")
            sentence.values = {
                "sentence": topic.values["content"],
                "context": topic.values["source"]
            }
            self.outputs = [sentence]
        except Exception as e:
            self.set_error(f"Error in bridge: {str(e)}")

def process_interview():
    # Configure LLM
    llm_config = LLMConfig(
        name="mistral:instruct",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9,
        num_ctx=4096
    )

    # Sample text
    sample_text = """
    My name is Maria. I am a professional caregiver. I am working in 'Angeli Custodi' since 5 years.
    My main responsibility is to provide social assistance to guests.
    My work is also to receive and talk with guests' relatives.
    We must handle guests' anxiety that is due to their desire to live normally,
    have social relationships and receive a good service.
    I use a computer to write the daily report. Sometimes I use to take notes in a block note during my shift,
    so I must to re-write all before leaving.
    """

    # Create input interview thought
    interview = InterviewThought("initial_interview")
    interview.values = {
        "text": sample_text,
        "source": "interview_20240329.txt"
    }

    # Step 1: Analyze interview into topics
    analyzer = InterviewAnalyzer("analyzer", llm_config)
    analyzer.process({"input": interview})

    if analyzer.has_error:
        print(f"Analyzer error: {analyzer.error_message}")
        return

    topic_thoughts = analyzer.outputs

    # Step 2: Process each topic
    decontextualized_thoughts = []
    for i, topic in enumerate(topic_thoughts):
        # Bridge to convert topic to sentence
        bridge = TopicToSentenceBridge(f"bridge_{i}", llm_config)
        bridge.process({"topic": topic})

        if bridge.has_error:
            print(f"Bridge error for topic {i}: {bridge.error_message}")
            continue

        # Decontextualize the sentence
        decontextualizer = Decontextualizer(f"decontext_{i}", llm_config)
        decontextualizer.process({"input": bridge.outputs[0]})

        if decontextualizer.has_error:
            print(f"Decontextualizer error for topic {i}: {decontextualizer.error_message}")
            continue

        decontextualized_thoughts.extend(decontextualizer.outputs)

    # Print results
    print("\nDecontextualized Results:")
    for i, thought in enumerate(decontextualized_thoughts, 1):
        print(f"\n{i}. Original: {thought.values['in_context_sentence']}")
        print(f"   Standalone: {thought.values['standalone_sentence']}")

if __name__ == "__main__":
    process_interview()
