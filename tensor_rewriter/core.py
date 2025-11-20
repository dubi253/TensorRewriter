from typing import List, Dict, Any, Tuple

class Tensor:
    def __init__(self, name: str, shape: Tuple[int, ...]):
        self.name = name
        self.shape = shape

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape})"

    def __eq__(self, other):
        return self.name == other.name and self.shape == other.shape

    def __hash__(self):
        return hash((self.name, self.shape))

class Operator:
    def __init__(self, op_type: str, inputs: List[Tensor], output: Tensor, params: Dict[str, Any] = None):
        self.op_type = op_type
        self.inputs = inputs
        self.output = output
        self.params = params if params else {}

    def __repr__(self):
        input_names = ", ".join([t.name for t in self.inputs])
        return f"{self.output.name} = {self.op_type}({input_names})"

class Graph:
    def __init__(self, inputs: List[Tensor], operators: List[Operator], outputs: List[Tensor]):
        self.inputs = inputs
        self.operators = operators
        self.outputs = outputs

    def add_operator(self, op: Operator):
        self.operators.append(op)

    def __repr__(self):
        ops_str = "\n".join([str(op) for op in self.operators])
        return f"Graph Inputs: {self.inputs}\n{ops_str}\nGraph Outputs: {self.outputs}"
