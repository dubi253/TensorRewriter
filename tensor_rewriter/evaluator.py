import numpy as np
from typing import Dict, List
from .core import Graph, Tensor
from .ops import OPS_REGISTRY

class Evaluator:
    def evaluate(self, graph: Graph, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Evaluates the graph with the given inputs.
        inputs: Dict mapping Tensor name to numpy array.
        Returns: Dict mapping Tensor name to numpy array for all tensors in the graph (or just outputs).
        """
        # Store all intermediate results
        context = inputs.copy()

        for op in graph.operators:
            # Gather inputs for this op
            op_inputs = []
            for inp in op.inputs:
                if inp.name not in context:
                    raise ValueError(f"Tensor {inp.name} not found in context. Graph might not be topologically sorted or input missing.")
                op_inputs.append(context[inp.name])
            
            # Compute
            op_impl = OPS_REGISTRY.get(op.op_type)
            if not op_impl:
                raise ValueError(f"Operator {op.op_type} not implemented.")
            
            output_val = op_impl.compute(op_inputs, op.params)
            context[op.output.name] = output_val

        # Return only the graph outputs
        result = {}
        for out in graph.outputs:
            result[out.name] = context[out.name]
        
        return result
