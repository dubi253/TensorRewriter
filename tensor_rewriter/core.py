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

    def get_structural_signature(self) -> str:
        """
        Generates a canonical string representation of the graph structure,
        abstracting away specific tensor names for intermediate nodes.
        """
        # Map tensor objects to canonical names
        # Inputs keep their names
        tensor_map = {t: t.name for t in self.inputs}
        
        sig_parts = []
        intermediate_counter = 0
        
        for op in self.operators:
            # Resolve input names
            input_keys = []
            for inp in op.inputs:
                if inp in tensor_map:
                    input_keys.append(tensor_map[inp])
                else:
                    # This should not happen if graph is valid and topological
                    # But if it does, maybe it's a constant or something not in inputs?
                    # For now, fallback to name
                    input_keys.append(inp.name)
            
            # Sort params for deterministic order
            params_str = ""
            if op.params:
                sorted_items = sorted(op.params.items())
                params_str = ",".join([f"{k}={v}" for k, v in sorted_items])
            
            op_sig = f"{op.op_type}({','.join(input_keys)})[{params_str}]"
            sig_parts.append(op_sig)
            
            # Assign canonical name to output
            # We use a deterministic counter
            canonical_out_name = f"%{intermediate_counter}"
            intermediate_counter += 1
            tensor_map[op.output] = canonical_out_name
            
        return "\n".join(sig_parts)
