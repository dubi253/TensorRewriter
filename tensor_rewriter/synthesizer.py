import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from .core import Graph, Tensor, Operator
from .ops import OPS_REGISTRY
from .evaluator import Evaluator

def execute_op_task(op_name, input_vals, input_shapes, params):
    """
    Standalone function for multiprocessing.
    Returns (output_val, output_shape) or None if failed.
    """
    op_cls = OPS_REGISTRY.get(op_name)
    if not op_cls: return None
    
    try:
        out_shape = op_cls.get_output_shape(input_shapes, params)
        out_val = op_cls.compute(input_vals, params)
        return (out_val, out_shape)
    except Exception as e:
        # print(f"Error executing {op_name}: {e}")
        return None

class Synthesizer:
    def __init__(self, input_config: Dict[str, Tuple[int, ...]], max_ops: int = 3, epsilon: float = 1e-4):
        self.input_config = input_config
        self.max_ops = max_ops
        self.epsilon = epsilon
        self.evaluator = Evaluator()
        
        # State
        self.tensors: List[Tensor] = [] # All available tensors
        self.tensor_to_creator: Dict[Tensor, Operator] = {} # Trace back
        self.tensor_values: Dict[Tensor, np.ndarray] = {} # Fingerprints
        self.fingerprint_map: Dict[bytes, List[Tensor]] = {} # Map hash of value to tensors
        
        # Rules: List of (Source Graph, Target Graph)
        self.rules = []

    def generate_random_inputs(self) -> Dict[str, np.ndarray]:
        inputs = {}
        for name, shape in self.input_config.items():
            inputs[name] = np.random.normal(0, 1, shape).astype(np.float32)
        return inputs

    def get_fingerprint(self, value: np.ndarray) -> bytes:
        # Simple hashing might be unstable with floats, so we round or use tolerance buckets
        # For now, let's just use the raw bytes of a rounded array
        rounded = np.round(value / self.epsilon) * self.epsilon
        return rounded.tobytes()

    def check_equivalence(self, t1: Tensor, val2: np.ndarray) -> bool:
        v1 = self.tensor_values[t1]
        # Use np.allclose for robust equivalence checking
        return np.allclose(v1, val2, rtol=1e-5, atol=self.epsilon)

    def reconstruct_graph(self, output_tensor: Tensor) -> Graph:
        # Backtrack to build the graph for this tensor
        ops = []
        inputs = []
        
        # Simple BFS or DFS to find all ops
        # We need to identify which tensors are "global inputs" vs "intermediate"
        
        visited_ops = set()
        pending_tensors = [output_tensor]
        graph_inputs = set()
        
        # We need to traverse backwards
        # But to build the graph object, we need the ops in topological order (forward)
        
        # Let's just collect all ancestors
        def collect(t: Tensor, op_list, seen_ops, seen_tensors):
            if t in seen_tensors: return
            seen_tensors.add(t)
            
            creator = self.tensor_to_creator.get(t)
            if creator:
                if creator in seen_ops: return
                seen_ops.add(creator)
                # Recurse on inputs
                for inp in creator.inputs:
                    collect(inp, op_list, seen_ops, seen_tensors)
                op_list.append(creator)
            else:
                # It's a global input
                graph_inputs.add(t)

        collected_ops = []
        collect(output_tensor, collected_ops, set(), set())
        
        # collected_ops should be in topological order because we append AFTER processing inputs
        
        return Graph(list(graph_inputs), collected_ops, [output_tensor])

    def synthesize(self):
        print("Initializing synthesis...")
        # 1. Initialize inputs
        input_data = self.generate_random_inputs()
        
        for name, shape in self.input_config.items():
            t = Tensor(name, shape)
            self.tensors.append(t)
            self.tensor_values[t] = input_data[name]
            # No creator for inputs
        
        # 2. Enumeration Loop
        print(f"Enumerating up to {self.max_ops} operators...")
        
        # We iterate by "depth" effectively by processing the list of tensors
        # But since we append to self.tensors, we need to be careful not to infinite loop or re-process too much
        # A simple approach: generations.
        
        current_tensors = list(self.tensors)
        
        for step in range(self.max_ops):
            print(f"Step {step + 1}/{self.max_ops}. Tensors available: {len(self.tensors)}")
            new_tensors = []
            
            # Collect tasks
            tasks = []
            
            for op_name, op_cls in OPS_REGISTRY.items():
                # 1-ary
                for t1 in self.tensors:
                    if op_cls.is_valid([t1.shape]):
                        # Get valid params
                        param_list = op_cls.get_valid_params([t1.shape])
                        for params in param_list:
                            tasks.append({
                                'op_name': op_name,
                                'op_cls': op_cls,
                                'inputs': [t1],
                                'params': params
                            })
                
                # 2-ary
                for t1 in self.tensors:
                    for t2 in self.tensors:
                        if op_cls.is_valid([t1.shape, t2.shape]):
                            param_list = op_cls.get_valid_params([t1.shape, t2.shape])
                            for params in param_list:
                                tasks.append({
                                    'op_name': op_name,
                                    'op_cls': op_cls,
                                    'inputs': [t1, t2],
                                    'params': params
                                })
            
            print(f"  Processing {len(tasks)} candidate operations...")
            
            # Execute tasks in parallel
            # We need to extract values to pass to worker
            futures = []
            with ProcessPoolExecutor() as executor:
                for task in tasks:
                    input_vals = [self.tensor_values[t] for t in task['inputs']]
                    input_shapes = [t.shape for t in task['inputs']]
                    future = executor.submit(execute_op_task, task['op_name'], input_vals, input_shapes, task['params'])
                    futures.append((task, future))
                
                for task, future in tqdm(futures, desc="Evaluating"):
                    result = future.result()
                    if result:
                        out_val, out_shape = result
                        self.process_result(task['op_name'], task['inputs'], task['params'], out_val, out_shape, new_tensors)

            # Add new tensors to the pool
            if not new_tensors:
                print("No new tensors generated. Stopping.")
                break
            
            self.tensors.extend(new_tensors)
            
        print(f"Synthesis complete. Found {len(self.rules)} rules.")

    def process_result(self, op_name, inputs, params, out_val, out_shape, new_tensors_list):
        # Create Tensor object
        new_name = f"t_{len(self.tensors) + len(new_tensors_list)}"
        out_tensor = Tensor(new_name, out_shape)
        
        # Create Operator object
        op = Operator(op_name, inputs, out_tensor, params)
        
        # Check for equivalence
        # 1. Check against existing tensors
        is_new = True
        for existing_t in self.tensors:
            if existing_t.shape == out_shape:
                if self.check_equivalence(existing_t, out_val): # We need to temporarily store value for out_tensor
                    # Found a rule!
                    # existing_t is equivalent to out_tensor
                    # But we only want to record it if the graphs are structurally different
                    # And we don't want to add out_tensor to the pool if it's redundant (pruning)
                    
                    # Construct graphs
                    g1 = self.reconstruct_graph(existing_t)
                    g2 = Graph(inputs, [op], [out_tensor]) # This is just the last step, we need full graph
                    
                    # Actually, reconstruct_graph needs to work on the new op too
                    # So we temporarily register it
                    self.tensor_to_creator[out_tensor] = op
                    g2_full = self.reconstruct_graph(out_tensor)
                    
                    # Simple structural check (string representation)
                    if str(g1) != str(g2_full):
                        # We want simplification rules: Complex -> Simple
                        # g1 is existing (simpler/older), g2_full is new (complex/newer)
                        self.rules.append((g2_full, g1))
                        # print(f"Found Rule: {g2_full}  ==>  {g1}")
                    
                    is_new = False
                    del self.tensor_to_creator[out_tensor] # Cleanup
                    break
        
        if is_new:
            # Add to list
            self.tensor_to_creator[out_tensor] = op
            self.tensor_values[out_tensor] = out_val
            new_tensors_list.append(out_tensor)



    
    def export_rules(self, filepath: str):
        import json
        
        def serialize_tensor(t):
            # Ensure shape is list of int, not numpy int
            shape = [int(d) for d in t.shape]
            return {"name": t.name, "shape": shape}
            
        def serialize_op(op):
            # Ensure params are serializable
            params = {}
            if op.params:
                for k, v in op.params.items():
                    if isinstance(v, (np.integer, int)):
                        params[k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        params[k] = float(v)
                    elif isinstance(v, (tuple, list, np.ndarray)):
                        params[k] = [int(x) if isinstance(x, (np.integer, int)) else x for x in v]
                    else:
                        params[k] = v
                        
            return {
                "type": op.op_type,
                "inputs": [t.name for t in op.inputs],
                "output": op.output.name,
                "output_shape": [int(d) for d in op.output.shape],
                "params": params
            }
            
        def serialize_graph(g):
            return {
                "inputs": [serialize_tensor(t) for t in g.inputs],
                "outputs": [serialize_tensor(t) for t in g.outputs],
                "operators": [serialize_op(op) for op in g.operators]
            }
            
        data = []
        for i, (g1, g2) in enumerate(self.rules):
            data.append({
                "id": i + 1,
                "source": serialize_graph(g1),
                "target": serialize_graph(g2)
            })
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
