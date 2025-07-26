from typing import Dict, Type
from src.got.node import GoTNode, LLMConfig
from src.got.thought import Thought
from .interview_analyzer import InterviewAnalyzer, InterviewThought, SentenceThought
from .decontextualizer import Decontextualizer, DecontextualizedThought


def process_interview():
    # LLM config
    llm_config = LLMConfig(
        name="llama2:7b",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9,
        num_ctx=4096
    )

    # Full input text (context for all sentences)
    sample_text = """
     My name is Maria. I am a professional caregiver and I have been working at 'Angeli Custodi' for the past five years.
    My primary responsibility is to offer social and emotional assistance to our elderly guests, many of whom feel isolated.
    """

    # Step 1: Analisi dell'intervista → frasi
    interview = InterviewThought("initial_interview")
    interview.values = {
        "text": sample_text,
        "source": "interview_20240329.txt"
    }

    analyzer = InterviewAnalyzer("sentence_splitter", llm_config)
    analyzer.process({"input": interview})

    if analyzer.has_error:
        print(f"Analyzer error: {analyzer.error_message}")
        return

    sentence_thoughts = analyzer.outputs

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Step 2: Decontestualizza ogni frase

    decontextualizer = Decontextualizer("shared_decontext", llm_config)

    decontextualized_thoughts = []
    for i, sent in enumerate(sentence_thoughts):
        # Costruiamo un nuovo SentenceThought con contesto completo
        enriched_sentence = SentenceThought(f"sent_{i}")
        enriched_sentence.values = {
            "sentence": sent.values["sentence"],
            "context": sample_text.strip(),  # contesto completo
            "title": sent.values.get("title", ""),
            "source": sent.values.get("context", "")
        }

        try:
            decontextualizer.process({"input": enriched_sentence})

            if decontextualizer.has_error:
                print(f"[ERROR] Decontext error at sentence {i}: {decontextualizer.error_message}")
                continue

            decontextualized_thoughts.extend(decontextualizer.outputs)
            print(f"[DEBUG] Done with sentence {i + 1}")

        except Exception as e:
            print(f"[EXCEPTION] Failed at sentence {i}: {e}")

    # Step 3: Output dei risultati
    print("\n=== Decontextualized Results ===")
    for i, thought in enumerate(decontextualized_thoughts, 1):
        print(f"\n{i}. Original:   {thought.values['in_context_sentence']}")
        print(f"   Standalone: {thought.values['standalone_sentence']}")
        print(f"   Title:      {thought.values.get('title', '')}")

    # Step 4: Lattice
    import os
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TAACO_main')))
    from TAACO_main.taaco_reticulator import TAACOReticulator

    # Step 4: Lattice
    import os
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TAACO_main')))
    from TAACO_main.taaco_reticulator import TAACOReticulator

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.join(base_dir, "taaco_input", "input")
    output_dir = os.path.join(base_dir, "taaco_output")
    lattice_output_file = os.path.join(base_dir, "output", "lattice_document.txt")

    reticulator = TAACOReticulator(input_dir, output_dir, llm_config)
    reticulator.build_reticle(decontextualized_thoughts)
    reticulator.generate_lattice_document(lattice_output_file)


if __name__ == "__main__":
    process_interview()
