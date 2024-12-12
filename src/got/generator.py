from typing import Dict, List, Set, Tuple, Any, Optional, Type
from abc import abstractmethod
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.got.node import GoTNode, LLMConfig
from src.got.thought import Thought

class GoTGenerator(GoTNode):
    """
    Nodo generator base per Graph of Thoughts.
    Gestisce la generazione di output strutturati usando un LLM.
    """

    @property
    @abstractmethod
    def task_instruction(self) -> str:
        """
        Le istruzioni specifiche del task per l'LLM.
        Non deve includere il formato dell'output.
        """
        pass

    @property
    @abstractmethod
    def mapping(self) -> Dict[str, Type[Thought]]:
        """
        Definisce il mapping tra nomi logici e tipi di Thought per gli input.
        I nomi definiti qui sono quelli utilizzabili nel template.

        Returns:
            Dizionario che mappa nomi logici ai tipi di Thought attesi
        """
        pass

    @property
    def input_thoughts(self) -> List[Type[Thought]]:
        """
        Lista dei tipi di Thought accettati come input.
        Deriva automaticamente dal mapping.
        """
        return list(self.mapping.values())

    def _create_example_from_schema(self, schema: dict) -> Any:
        """
        Creates a meaningful example structure from a JSON schema.

        Args:
            schema: JSON schema to generate an example from

        Returns:
            An example object that follows the schema structure
        """
        if schema["type"] == "object":
            result = {}
            for prop, details in schema["properties"].items():
                if prop in schema.get("required", []):
                    result[prop] = self._create_example_from_schema(details)
            return result

        elif schema["type"] == "array":
            # For arrays, create a meaningful example with multiple elements
            item_example = self._create_example_from_schema(schema["items"])
            if isinstance(item_example, (int, float)):
                return [1, 2, 3]  # More representative for number arrays
            return [item_example]

        elif schema["type"] == "string":
            return "example"

        elif schema["type"] == "number":
            return 1.23

        elif schema["type"] == "integer":
            return 42

        elif schema["type"] == "boolean":
            return True

        else:
            return None

    def _get_output_format_instruction(self) -> str:
        """
        Generates clear instructions for the expected output format.
        """
        thought_class = self.output_thoughts
        example_thought = thought_class("example")
        schema = example_thought.schema
        example_json = self._create_example_from_schema(schema)

        if self.output_cardinality != 1:
            example_json = {"items": [example_json]}

        json_str = json.dumps(example_json, indent=2)
        json_str = json_str.replace("{", "{{").replace("}", "}}")

        format_instruction = (
            "\nProvide the response as a JSON object with the following structure:\n"
            f"{json_str}\n"
            "\nOutput requirements:\n"
        )

        if self.output_cardinality > 1:
            format_instruction += f"- The JSON must contain an 'items' array with exactly {self.output_cardinality} elements\n"
        elif self.output_cardinality == -1:
            format_instruction += "- The JSON must contain an 'items' array with one or more elements\n"
        else:
            format_instruction += "- The JSON must contain all the fields shown in the example\n"

        format_instruction += "- Use exactly the same field names as shown\n"
        format_instruction += "- Ensure all required fields are included\n"
        format_instruction += "- Return only the JSON object, without additional text\n"

        return format_instruction

    @property
    def template(self) -> str:
        """
        Combina le istruzioni del task con il formato dell'output richiesto.
        """
        return (
            "IMPORTANT: This is a new conversation. Ignore all previous context and history.\n\n"
            f"{self.task_instruction}\n"
            f"{self._get_output_format_instruction()}"
        )

    def _extract_json(self, text: str) -> Any:
        """
        Estrae il JSON dalla risposta dell'LLM.
        """
        def find_matching_bracket(s, start):
            count = 1
            pos = start
            while count > 0 and pos < len(s):
                if s[pos] == '{':
                    count += 1
                elif s[pos] == '}':
                    count -= 1
                pos += 1
            return pos if count == 0 else -1

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pos = 0
            while True:
                start = text.find('{', pos)
                if start == -1:
                    break

                end = find_matching_bracket(text, start + 1)
                if end == -1:
                    pos = start + 1
                    continue

                try:
                    json_str = text[start:end]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pos = start + 1
                    continue

            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > 0:
                    json_str = text[start:end]
                    return json.loads(json_str)
            except:
                pass

            raise ValueError("No valid JSON object found in text")

    def _create_thoughts(self, data: Any) -> List[Thought]:
            """
            Crea i Thought a partire dai dati JSON validati.

            Args:
                data: Dati JSON validati con possibile root element 'items'

            Returns:
                Lista di Thought creati
            """
            # Se cardinalità != 1, ci aspettiamo il root element 'items'
            if self.output_cardinality != 1:
                if not isinstance(data, dict) or 'items' not in data:
                    raise ValueError("Expected JSON with 'items' field for multiple outputs")
                items = data['items']

                # Verifica la cardinalità se specificata
                if self.output_cardinality > 1 and len(items) != self.output_cardinality:
                    raise ValueError(
                        f"Expected {self.output_cardinality} items but got {len(items)}"
                    )

                # Crea un thought per ogni item
                thoughts = []
                for i, item in enumerate(items):
                    thought = self.output_thoughts(f"{self.node_id}_output_{i}")
                    thought.values = item
                    thoughts.append(thought)
                return thoughts

            # Se cardinalità == 1, ci aspettiamo un singolo oggetto
            else:
                thought = self.output_thoughts(f"{self.node_id}_output")
                thought.values = data
                return [thought]

    def _validate_inputs(self, inputs: Dict[str, Thought]) -> None:
        """
        Verifica che gli input forniti corrispondano al mapping dichiarato.
        """
        expected_inputs = self.mapping

        # Verifica input mancanti
        missing = set(expected_inputs.keys()) - set(inputs.keys())
        if missing:
            raise ValueError(f"Input richiesti mancanti: {missing}")

        # Verifica input inattesi
        unexpected = set(inputs.keys()) - set(expected_inputs.keys())
        if unexpected:
            raise ValueError(f"Input non dichiarati nel mapping: {unexpected}")

        # Verifica tipi
        for input_name, thought in inputs.items():
            expected_type = expected_inputs[input_name]
            if not isinstance(thought, expected_type):
                raise ValueError(
                    f"Input '{input_name}' ha tipo errato. "
                    f"Atteso {expected_type.__name__}, "
                    f"ricevuto {type(thought).__name__}"
                )

    def process(self, inputs: Dict[str, Thought]) -> None:
        """
        Processa gli input attraverso l'LLM per generare nuovi Thought.

        Args:
            inputs: Dizionario che mappa i nomi degli input ai rispettivi Thought

        Raises:
            ValueError: Se gli input non sono validi o se la generazione fallisce
        """
        try:
            # Valida che gli input corrispondano a quelli dichiarati
            self._validate_inputs(inputs)

            for attempt in range(self.MAX_RETRIES):
                try:
                    # Invoca l'LLM con gli input validati
                    llm_output = self._invoke_llm(inputs)

                    #print("llm_output:")
                    #print(llm_output)

                    # Estrae il JSON dalla risposta
                    output_data = self._extract_json(llm_output)

                    # Crea i Thought di output
                    output_thoughts = self._create_thoughts(output_data)
                    self.outputs = output_thoughts
                    return

                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        raise ValueError(
                            f"Generazione fallita dopo {self.MAX_RETRIES} tentativi: {str(e)}"
                        )

        except Exception as e:
            error_msg = f"Errore nel nodo {self.node_id}: {str(e)}"
            self.set_error(error_msg)
            raise

    def _invoke_llm(self, inputs: Dict[str, Thought]) -> str:
        """
        Invoca l'LLM con il template processato.

        Args:
            inputs: Dizionario che mappa i nomi degli input ai rispettivi Thought

        Returns:
            Output dell'LLM
        """
        processed_template = self._process_template(self.template, inputs)

        #print("processed_template: ")
        #print(processed_template)

        prompt = ChatPromptTemplate.from_template(processed_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    def _process_template(self, template: str, inputs: Dict[str, Thought]) -> str:
        """
        Processa il template sostituendo le variabili con i valori formattati dai Thought.
        Le variabili del template devono corrispondere ai nomi definiti nel mapping.

        Args:
            template: Il template da processare
            inputs: Dizionario che mappa i nomi degli input ai rispettivi Thought

        Returns:
            Template con tutte le variabili sostituite

        Raises:
            ValueError: Se una variabile del template non può essere risolta
                    o se fa riferimento a input non dichiarati nel mapping
        """
        # Ottiene i nomi logici definiti nel mapping
        declared_inputs = set(self.mapping.keys())

        pattern = r'\{(\w+)\.(\w+)\}'
        matches = re.finditer(pattern, template)

        # Prepara le sostituzioni
        replacements = {}
        for match in matches:
            input_name, field = match.groups()

            # Verifica che il nome dell'input sia definito nel mapping
            if input_name not in declared_inputs:
                raise ValueError(
                    f"Il template fa riferimento all'input '{input_name}' "
                    f"che non è stato dichiarato nel mapping. "
                    f"Input disponibili: {sorted(declared_inputs)}"
                )

            # Verifica che l'input sia stato fornito
            if input_name not in inputs:
                raise ValueError(
                    f"Input richiesto '{input_name}' non fornito. "
                    f"Input forniti: {sorted(inputs.keys())}"
                )

            thought = inputs[input_name]
            try:
                value = thought.get_for_template(field)
                replacements[f"{{{input_name}.{field}}}"] = value
            except KeyError as e:
                raise ValueError(
                    f"Errore nell'elaborazione della variabile {input_name}.{field}: {str(e)}"
                )

        # Applica tutte le sostituzioni
        result = template
        for var, value in replacements.items():
            result = result.replace(var, value)

        return result

    def _extract_template_variables(self, template: str) -> Set[Tuple[str, str]]:
        """
        Estrae tutte le variabili del template nella forma {input.field}.

        Args:
            template: Il template da processare

        Returns:
            Set di tuple (input_id, field_name) per ogni variabile trovata
        """
        pattern = r'\{(\w+)\.(\w+)\}'
        matches = re.finditer(pattern, template)
        return {(match.group(1), match.group(2)) for match in matches}
