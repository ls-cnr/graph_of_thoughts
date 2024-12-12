from typing import List, Dict, Type, Any
from typing import cast


from src.got.node import LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from src.got.keepbest import GoTKeepBest
from src.got.repeat import GoTRepeat
from src.operations.graph import GraphOfOperations


class IntSetThought(Thought):
    """Thought che rappresenta un insieme ordinato di interi"""

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["values", "size"],
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "size": {
                    "type": "integer",
                    "minimum": 0
                }
            },
            "additionalProperties": False
        }

    def get_for_template(self, key: str) -> str:
        """
        Fornisce una rappresentazione formattata dei valori per il template.

        Args:
            key: Il campo richiesto

        Returns:
            str: Il valore formattato per il template

        Raises:
            KeyError: Se la chiave richiesta non è valida
        """
        match key:
            case "values":
                return ", ".join(map(str, self._values["values"]))
            case "size":
                return str(self._values["size"])
            case "halfsize":
                return str(self._values["size"]/2)
            case _:
                raise KeyError(f"Campo '{key}' non valido per IntSetThought")

    @classmethod
    def create(cls, thought_id: str, values: List[int]) -> 'IntSetThought':
        """
        Factory method per creare un IntSetThought.

        Args:
            thought_id: Identificatore del thought
            values: Lista di interi

        Returns:
            IntSetThought inizializzato con i valori forniti
        """
        thought = cls(thought_id)
        thought.values = {
            "values": values,
            "size": len(values)
        }
        return thought

    def get_values(self) -> List[int]:
        """Restituisce la lista di valori contenuta nel thought"""
        return self.values["values"]



class Splitter(GoTGenerator):
    """Generator che divide un IntSet in due sottoinsiemi di uguale dimensione"""

    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input": IntSetThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return IntSetThought

    @property
    def output_cardinality(self) -> int:
        return 2  # Produce sempre due sottoinsiemi

    @property
    def task_instruction(self) -> str:
        return """
        Task: Split the given list of integers into two equal-sized subsets.

        Input list: {input.values}

        <Instructions>
        1. Create exactly two subsets of equal size ({input.halfsize} integers each)
        2. Each number from the original list must appear exactly once in either subset
        3. Each subset must be a valid list of integers
        4. The order of numbers within each subset can be arbitrary
        5. All numbers from the original list must be used
        <\Instructions>
        """

class Sorter(GoTGenerator):
    """Generator that sorts a set of integers in ascending order"""

    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {"input": IntSetThought}

    @property
    def output_thoughts(self) -> Type[Thought]:
        return IntSetThought

    @property
    def output_cardinality(self) -> int:
        return 1

    @property
    def task_instruction(self) -> str:
        return """
        Task: Sort the following array of integers in ascending order (smallest to largest).

        Input array: {input.values}
        Number of elements: {input.size}

        <Instructions>
        1. Arrange these numbers in ascending order
        2. Include every number exactly once
        3. Return the sorted array in the specified JSON format
        <\Instructions>
        """

class Merger(GoTGenerator):
    """Generator che unisce due IntSet in un unico IntSet"""

    @property
    def mapping(self) -> Dict[str, Type[Thought]]:
        return {
            "input1": IntSetThought,
            "input2": IntSetThought
        }

    @property
    def output_thoughts(self) -> Type[Thought]:
        return IntSetThought

    @property
    def output_cardinality(self) -> int:
        return 1  # Produce un singolo insieme unito

    @property
    def task_instruction(self) -> str:
        return """
        Combine two sets of integers into a single unified set.

        First set of integers: {input1.values}
        Second set of integers: {input2.values}

        <Instruction>
        Your task is to merge these sets while adhering to these requirements:
        1. Include every element from both input sets in the result
        2. Preserve the relative ordering of elements from each input set
        3. Ensure no elements are duplicated or omitted in the final set
        </Instruction>
        """


class SortingKeepBest(GoTKeepBest):
    """Selettore che sceglie il miglior ordinamento tra quelli proposti"""

    @property
    def input_thoughts(self) -> List[Type[Thought]]:
        return [IntSetThought]

    def _assign_scores(self, input: Thought) -> Dict[str, float]:
        """
        Assegna uno score all'ordinamento del thought.

        Args:
            thought: IntSetThought da valutare

        Returns:
            Dict[str, float]: Score dell'ordinamento normalizzato nell'intervallo [0,1]
        """
        int_thought = cast(IntSetThought, input)
        values = int_thought.get_values()
        size = len(values)

        if size <= 1:
            return {"ordering_score": 1.0}

        # Calcola la percentuale di coppie correttamente ordinate
        correct_pairs = sum(
            1 for i in range(size-1)
            if values[i] <= values[i+1]
        )
        max_pairs = size - 1

        ordering_score = correct_pairs / max_pairs
        return {"ordering_score": ordering_score}

    def _compare(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> int:
        """
        Confronta due set di score per determinare il miglior ordinamento.

        Args:
            scores1: Score del primo thought
            scores2: Score del secondo thought

        Returns:
            int: >0 se scores1 è migliore, <0 se scores2 è migliore, 0 se equivalenti
        """
        score1 = scores1["ordering_score"]
        score2 = scores2["ordering_score"]

        # Converti la differenza in un intero, mantenendo il segno
        return int((score1 - score2) * 1000)

def create_sorting_graph(llm_config: LLMConfig) -> GraphOfOperations:
    """
    Crea un grafo di operazioni per l'ordinamento parallelo di insiemi.

    Args:
        llm_config: Configurazione del modello LLM da utilizzare

    Returns:
        GraphOfOperations configurato secondo lo schema richiesto
    """
    # Crea il grafo principale
    graph = GraphOfOperations("sorting_graph", llm_config)

    # Crea i nodi
    splitter = Splitter("splitter", llm_config)

    # Branch 1
    sorter1 = Sorter("sorter1", llm_config)
    repeater1 = GoTRepeat("repeater1", llm_config, sorter1, k=2)
    keeper1 = SortingKeepBest("keeper1", llm_config)

    # Branch 2
    sorter2 = Sorter("sorter2", llm_config)
    repeater2 = GoTRepeat("repeater2", llm_config, sorter2, k=2)
    keeper2 = SortingKeepBest("keeper2", llm_config)

    # Nodo finale
    merger = Merger("merger", llm_config)

    # Aggiungi i nodi al grafo
    graph.add_node(splitter, is_input=True)

    graph.add_node(repeater1)
    graph.add_node(keeper1)

    graph.add_node(repeater2)
    graph.add_node(keeper2)

    graph.add_node(merger, is_output=True)

    # Definisci le connessioni
    # Il Splitter produce due output che vengono referenziati come "output"
    graph.add_edge("splitter", "repeater1", "output", "input")
    graph.add_edge("splitter", "repeater2", "output", "input")

    # I GoTRepeat producono gli output che vengono passati ai rispettivi Keeper
    graph.add_edge("repeater1", "keeper1", "output", "input")
    graph.add_edge("repeater2", "keeper2", "output", "input")

    # I Keeper inviano i loro output al Merger
    graph.add_edge("keeper1", "merger", "output", "input1")
    graph.add_edge("keeper2", "merger", "output", "input2")

    return graph

def test_graph():
    # Configura l'LLM
    config = LLMConfig(
        name="mistral:instruct",
        temperature=0.1
    )

    # Crea il grafo
    graph = create_sorting_graph(config)

    # Prepara l'input
    input_set = IntSetThought.create(
        "input",
        [45, 12, 89, 33, 56, 71, 24, 90, 15, 68,
         92, 47, 31, 85, 19, 73, 58, 26, 64, 37]
    )

    # Esegui il grafo
    graph.process({"input": input_set})

    # Verifica l'output
    if not graph.has_error:
        result = graph.outputs[0]
        int_thought = cast(IntSetThought, result)
        print("Insieme ordinato:", int_thought.get_values())
    else:
        print("Errore:", graph.error_message)


def test_splitter():
    """Test the functionality of the Splitter node."""

    # Configure the LLM
    config = LLMConfig(
        name="mistral:instruct",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9
    )

    # Create a Splitter instance
    splitter = Splitter("test_splitter", config)

    # Create input data
    input_values = [45, 12, 89, 33, 56, 71, 24, 90, 15, 68]
    input_thought = IntSetThought.create("input", input_values)

    print("\nTesting Splitter Node")
    print("-" * 50)
    print(f"\nInput set: {input_thought.get_values()}")
    print(f"Input size: {input_thought.values['size']}")

    # Process the input
    try:
        splitter.process({"input": input_thought})

        if not splitter.has_error:
            print("\nSplit successful!")
            print("\nFirst subset:")
            subset1 = splitter.outputs[0]
            print(f"Values: {subset1.get_values()}")
            print(f"Size: {subset1.values['size']}")

            print("\nSecond subset:")
            subset2 = splitter.outputs[1]
            print(f"Values: {subset2.get_values()}")
            print(f"Size: {subset2.values['size']}")

            # Validate the split
            all_values = set(subset1.get_values() + subset2.get_values())
            original_values = set(input_values)

            print("\nValidation:")
            print(f"All original elements present: {all_values == original_values}")
            print(f"Correct sizes: {subset1.values['size'] == subset2.values['size'] == len(input_values)//2}")

        else:
            print(f"\nError occurred: {splitter.error_message}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

def verify_sorting(original: List[int], sorted_list: List[int]) -> bool:
    """
    Verifies that the sorted output is correct.

    Args:
        original: The original list of integers
        sorted_list: The supposedly sorted list to verify

    Returns:
        bool: True if the sorting is correct
    """
    # Check if lists have the same elements
    if set(original) != set(sorted_list):
        return False

    # Check if list is actually sorted
    return all(sorted_list[i] <= sorted_list[i+1] for i in range(len(sorted_list)-1))

def test_sorter():
    """Test the functionality of the Sorter node."""

    config = LLMConfig(
        name="mistral:instruct",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9
    )

    sorter = Sorter("test_sorter", config)

    # Create test input with deliberately unsorted values
    input_values = [45, 12, 89, 33, 56, 71, 24, 90, 15, 68]
    input_thought = IntSetThought.create("input", input_values)

    print("\nTesting Sorter Node")
    print("-" * 50)
    print(f"\nInput set: {input_thought.get_values()}")
    print(f"Expected order: {sorted(input_values)}")

    try:
        sorter.process({"input": input_thought})

        if not sorter.has_error:
            result = cast(IntSetThought, sorter.outputs[0])
            sorted_values = result.get_values()

            print("\nSorting result:")
            print(f"Output values: {sorted_values}")
            print(f"Output size: {result.values['size']}")

            # Perform validation
            is_valid = verify_sorting(input_values, sorted_values)
            print("\nValidation:")
            print(f"Size preserved: {len(input_values) == len(sorted_values)}")
            print(f"All elements preserved: {set(input_values) == set(sorted_values)}")
            print(f"Correctly sorted: {is_valid}")

            # Additional validation details if there are issues
            if not is_valid:
                print("\nDetailed validation:")
                print("Elements in original but not in result:",
                      set(input_values) - set(sorted_values))
                print("Elements in result but not in original:",
                      set(sorted_values) - set(input_values))

                # Check if consecutive pairs are properly ordered
                if len(sorted_values) > 1:
                    incorrect_pairs = [
                        (sorted_values[i], sorted_values[i+1])
                        for i in range(len(sorted_values)-1)
                        if sorted_values[i] > sorted_values[i+1]
                    ]
                    if incorrect_pairs:
                        print("Incorrectly ordered pairs:", incorrect_pairs)

        else:
            print(f"\nError occurred: {sorter.error_message}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

def verify_merger_result(set1: List[int], set2: List[int], merged: List[int]) -> bool:
    """
    Verifies that the merger result contains all elements from both input sets.

    Args:
        set1: First input set
        set2: Second input set
        merged: Result of merging set1 and set2

    Returns:
        bool: True if the merger is correct
    """
    original_elements = set(set1 + set2)
    merged_elements = set(merged)
    return original_elements == merged_elements

def test_merger():
    """Tests the functionality of the Merger node."""

    config = LLMConfig(
        name="llama2",
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9
    )

    merger = Merger("test_merger", config)

    # Create two test input sets
    input_set1 = [12, 24, 33, 45, 56]
    input_set2 = [15, 68, 71, 89, 90]

    thought1 = IntSetThought.create("input1", input_set1)
    thought2 = IntSetThought.create("input2", input_set2)

    print("\nTesting Merger Node")
    print("-" * 50)
    print(f"\nFirst input set: {input_set1}")
    print(f"Second input set: {input_set2}")
    print(f"Total elements: {len(input_set1) + len(input_set2)}")

    try:
        merger.process({"input1": thought1, "input2": thought2})

        if not merger.has_error:
            result = cast(IntSetThought, merger.outputs[0])
            merged_values = result.get_values()

            print("\nMerger result:")
            print(f"Output values: {merged_values}")
            print(f"Output size: {result.values['size']}")

            print("\nValidation:")
            print(f"Size correct: {len(merged_values) == len(input_set1) + len(input_set2)}")

            is_valid = verify_merger_result(input_set1, input_set2, merged_values)
            print(f"All elements preserved: {is_valid}")

            if not is_valid:
                original_elements = set(input_set1 + input_set2)
                merged_elements = set(merged_values)

                print("\nDetailed validation:")
                print("Elements missing from result:",
                      original_elements - merged_elements)
                print("Unexpected elements in result:",
                      merged_elements - original_elements)

        else:
            print(f"\nError occurred: {merger.error_message}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

# Esempio di utilizzo
if __name__ == "__main__":
    test_merger()
