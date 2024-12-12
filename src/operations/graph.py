from typing import Dict, List, Set, Tuple, Type, Optional
from dataclasses import dataclass
from collections import defaultdict

from src.got.node import GoTNode,LLMConfig
from src.got.thought import Thought
from src.got.generator import GoTGenerator
from src.got.keepbest import GoTKeepBest
from src.got.adapter import GoTAdapter


@dataclass
class Edge:
    """Rappresenta un collegamento tra nodi nel grafo."""
    from_node: str  # ID del nodo sorgente
    to_node: str    # ID del nodo destinazione
    from_output: str  # Nome dell'output dal nodo sorgente
    to_input: str     # Nome dell'input al nodo destinazione

class GraphOfOperations(GoTNode):
    """
    Implementa un grafo di operazioni come composizione ricorsiva di GoTNode.
    Ogni nodo nel grafo è un GoTNode e gli archi rappresentano le dipendenze tra i nodi.
    """

    def __init__(self, node_id: str, llm_config: LLMConfig):
        super().__init__(node_id, llm_config)

        # Struttura del grafo
        self.nodes: Dict[str, GoTNode] = {}  # node_id -> node
        self.edges: List[Edge] = []  # Collegamenti tra nodi

        # Nodi di input/output del grafo
        self.input_nodes: Set[str] = set()  # IDs dei nodi che ricevono input esterni
        self.output_nodes: Set[str] = set() # IDs dei nodi che producono output finali

        # Cache per i risultati intermedi
        self._node_outputs: Dict[str, List[Thought]] = {}

    def add_node(self, node: GoTNode, is_input: bool = False, is_output: bool = False) -> None:
        """
        Aggiunge un nodo al grafo.

        Args:
            node: Il nodo da aggiungere
            is_input: True se il nodo riceve input esterni
            is_output: True se il nodo produce output finali
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with id {node.node_id} already exists")

        self.nodes[node.node_id] = node

        if is_input:
            self.input_nodes.add(node.node_id)
        if is_output:
            self.output_nodes.add(node.node_id)

    def add_edge(self, from_node: str, to_node: str,
                from_output: str = "output", to_input: str = "input") -> None:
        """
        Aggiunge un collegamento tra due nodi.

        Args:
            from_node: ID del nodo sorgente
            to_node: ID del nodo destinazione
            from_output: Nome dell'output dal nodo sorgente
            to_input: Nome dell'input al nodo destinazione
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")

        edge = Edge(from_node, to_node, from_output, to_input)
        self.edges.append(edge)

    def _get_node_dependencies(self) -> Dict[str, Set[str]]:
        """
        Calcola le dipendenze tra i nodi.

        Returns:
            Dizionario node_id -> set di node_id da cui dipende
        """
        dependencies = defaultdict(set)
        for edge in self.edges:
            dependencies[edge.to_node].add(edge.from_node)
        return dependencies

    def _get_ready_nodes(self, dependencies: Dict[str, Set[str]],
                        completed_nodes: Set[str]) -> List[str]:
        """
        Trova i nodi pronti per essere eseguiti.
        Un nodo è pronto quando tutti i suoi predecessori sono stati completati.

        Returns:
            Lista di node_id pronti per l'esecuzione
        """
        ready_nodes = []
        for node_id in self.nodes:
            if node_id not in completed_nodes:
                node_deps = dependencies.get(node_id, set())
                if all(dep in completed_nodes for dep in node_deps):
                    ready_nodes.append(node_id)
        return ready_nodes

    def _prepare_node_inputs(self, node_id: str) -> Dict[str, Thought]:
        """
        Prepara gli input per un nodo dai risultati dei suoi predecessori.

        Args:
            node_id: ID del nodo per cui preparare gli input

        Returns:
            Dizionario degli input per il nodo
        """
        inputs = {}
        input_edges = [e for e in self.edges if e.to_node == node_id]

        for edge in input_edges:
            from_node = self.nodes[edge.from_node]
            target_node = self.nodes[node_id]
            if isinstance(from_node, GoTGenerator) and isinstance(target_node, GoTKeepBest):
                # Usa l'adapter per collegamenti Generator -> Keeper
                GoTAdapter.connect_generator_to_keeper(
                    generator=from_node,
                    keeper=target_node
                )
                # Gli input verranno gestiti internamente dall'adapter
                return {}
            else:
                # Gestione standard degli input
                predecessor_outputs = self._node_outputs[edge.from_node]
                inputs[edge.to_input] = predecessor_outputs[0]  # Prende il primo output

        return inputs

    def process(self, inputs: Dict[str, Thought]) -> None:
        """
        Esegue il grafo di operazioni con output verboso.

        Args:
            inputs: Dizionario degli input esterni per i nodi di input
        """
        try:
            print(f"\n{'='*50}")
            print(f"Iniziando l'esecuzione del grafo: {self.node_id}")
            print(f"{'='*50}")

            print("\nInput ricevuti:")
            for input_id, thought in inputs.items():
                print(f"\nThought '{input_id}':")
                print(f"  Tipo: {type(thought).__name__}")
                print(f"  Valori: {thought.values}")

            # Resetta la cache dei risultati
            self._node_outputs.clear()

            # Distribuisce gli input esterni ai nodi di input
            print("\nElaborazione nodi di input:")
            print("-" * 30)
            for node_id in self.input_nodes:
                print(f"\nProcessando nodo di input: {node_id}")
                self.nodes[node_id].process(inputs)
                self._node_outputs[node_id] = self.nodes[node_id].outputs
                print(f"Output generati da {node_id}:")
                for i, thought in enumerate(self._node_outputs[node_id]):
                    print(f"  Output {i+1}:")
                    print(f"    Tipo: {type(thought).__name__}")
                    print(f"    Valori: {thought.values}")

            # Calcola le dipendenze tra i nodi
            dependencies = self._get_node_dependencies()
            completed_nodes = set(self.input_nodes)

            # Esegue i nodi in ordine topologico
            print("\nElaborazione nodi interni:")
            print("-" * 30)
            while len(completed_nodes) < len(self.nodes):
                ready_nodes = self._get_ready_nodes(dependencies, completed_nodes)

                if not ready_nodes:
                    raise ValueError("Rilevata dipendenza circolare nel grafo")

                for node_id in ready_nodes:
                    print(f"\nProcessando nodo: {node_id}")
                    node = self.nodes[node_id]
                    node_inputs = self._prepare_node_inputs(node_id)

                    print("Input preparati:")
                    for input_id, thought in node_inputs.items():
                        print(f"  Input '{input_id}':")
                        print(f"    Tipo: {type(thought).__name__}")
                        print(f"    Valori: {thought.values}")

                    if not node.outputs:
                        node.process(node_inputs)

                    if node.has_error:
                        print(f"ERRORE nel nodo {node_id}: {node.error_message}")
                        raise ValueError(f"Fallimento del nodo {node_id}: {node.error_message}")

                    self._node_outputs[node_id] = node.outputs
                    print(f"\nOutput generati da {node_id}:")
                    for i, thought in enumerate(self._node_outputs[node_id]):
                        print(f"  Output {i+1}:")
                        print(f"    Tipo: {type(thought).__name__}")
                        print(f"    Valori: {thought.values}")

                    completed_nodes.add(node_id)
                    print(f"Completato nodo {node_id}")

            # Raccoglie gli output finali
            print("\nRaccolta output finali:")
            print("-" * 30)
            final_outputs = []
            for output_node_id in self.output_nodes:
                print(f"\nRaccogliendo output da: {output_node_id}")
                final_outputs.extend(self._node_outputs[output_node_id])
            self.outputs = final_outputs

            print("\nOutput finali del grafo:")
            for i, thought in enumerate(self.outputs):
                print(f"  Output {i+1}:")
                print(f"    Tipo: {type(thought).__name__}")
                print(f"    Valori: {thought.values}")

            print(f"\n{'='*50}")
            print("Esecuzione del grafo completata con successo")
            print(f"{'='*50}\n")

        except Exception as e:
            error_msg = f"Errore nel grafo {self.node_id}: {str(e)}"
            print(f"\nERRORE: {error_msg}")
            self.set_error(error_msg)
            raise

    @property
    def input_thoughts(self) -> Dict[str, Type[Thought]]:
        """Tipi di Thought accettati come input dai nodi di input."""
        input_types = []
        for node_id in self.input_nodes:
            input_types.extend(self.nodes[node_id].input_thoughts)
        return input_types

    @property
    def output_thoughts(self) -> Type[Thought]:
        """Tipo di Thought prodotto come output dai nodi di output."""
        if not self.output_nodes:
            raise ValueError("No output nodes defined")
        # Per semplicità, assumiamo che tutti i nodi di output producano lo stesso tipo
        return self.nodes[list(self.output_nodes)[0]].output_thoughts

    @property
    def output_cardinality(self) -> int:
        """Cardinalità totale degli output prodotti dai nodi di output."""
        return sum(self.nodes[node_id].output_cardinality
                  for node_id in self.output_nodes)
