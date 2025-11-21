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
    def __init__(self, input_config: Dict[str, Tuple[int, ...]], max_ops: int = 3, epsilon: float = 1e-4, verification_runs: int = 0):
        self.input_config = input_config
        self.max_ops = max_ops
        self.epsilon = epsilon
        self.verification_runs = verification_runs
        self.evaluator = Evaluator()
        
        # Constants
        self.constants = [0.0, 1.0, -1.0]
        
        # State
        self.tensors: List[Tensor] = [] # All available tensors
        self.tensor_to_creator: Dict[Tensor, Operator] = {} # Trace back
        self.tensor_values: Dict[Tensor, np.ndarray] = {} # Fingerprints
        self.fingerprint_map: Dict[bytes, List[Tensor]] = {} # Map hash of value to tensors
        
        # Rules: List of (Source Graph, Target Graph)
        self.rules = []

    def get_constants_dict(self) -> Dict[str, np.ndarray]:
        return {f"Const({c})": np.array(c, dtype=np.float32) for c in self.constants}

    def generate_random_inputs(self) -> Dict[str, np.ndarray]:
        inputs = {}
        for name, shape in self.input_config.items():
            inputs[name] = np.random.normal(0, 1, shape).astype(np.float32)
        return inputs

    def get_fingerprint(self, value: np.ndarray) -> bytes:
        # Downsample for large arrays to improve performance
        flat = value.flatten()
        if flat.size > 100:
            # Deterministic sampling
            indices = np.linspace(0, flat.size - 1, 100).astype(int)
            sample = flat[indices]
        else:
            sample = flat
            
        # Simple hashing might be unstable with floats, so we round or use tolerance buckets
        rounded = np.round(sample / self.epsilon) * self.epsilon
        return rounded.tobytes()

    def check_equivalence(self, t1: Tensor, val2: np.ndarray) -> bool:
        v1 = self.tensor_values[t1]
        # Use np.allclose for robust equivalence checking
        return np.allclose(v1, val2, rtol=1e-5, atol=self.epsilon)

    def verify_equivalence(self, g1: Graph, g2: Graph) -> bool:
        if self.verification_runs <= 0:
            return True
            
        for _ in range(self.verification_runs):
            inputs = self.generate_random_inputs()
            inputs.update(self.get_constants_dict())
            try:
                res1 = self.evaluator.evaluate(g1, inputs)
                res2 = self.evaluator.evaluate(g2, inputs)
                
                # Get output values. Graphs might have multiple outputs but here we focus on the single tensor being synthesized
                # g1 and g2 are reconstructed for a specific output tensor.
                # reconstruct_graph returns [output_tensor] as outputs.
                
                val1 = res1[g1.outputs[0].name]
                val2 = res2[g2.outputs[0].name]
                
                if not np.allclose(val1, val2, rtol=1e-5, atol=self.epsilon):
                    return False
            except Exception as e:
                print(f"Verification failed with error: {e}")
                return False
                
        return True

    def instantiate_graph(self, template_graph: Graph, new_input_shapes: Dict[str, Tuple[int, ...]]) -> Graph | None:
        old_to_new: Dict[Tensor, Tensor] = {}
        new_inputs = []
        
        # 1. Setup Inputs
        for old_inp in template_graph.inputs:
            shape = new_input_shapes.get(old_inp.name)
            if shape is None: return None
            new_inp = Tensor(old_inp.name, shape)
            old_to_new[old_inp] = new_inp
            new_inputs.append(new_inp)
            
        new_ops = []
        
        # 2. Replay Operators
        for old_op in template_graph.operators:
            # Get new inputs
            new_op_inputs = []
            new_op_input_shapes = []
            for old_inp in old_op.inputs:
                if old_inp not in old_to_new:
                    return None # Should not happen in valid graph
                new_t = old_to_new[old_inp]
                new_op_inputs.append(new_t)
                new_op_input_shapes.append(new_t.shape)
            
            # Check validity
            op_cls = OPS_REGISTRY.get(old_op.op_type)
            if not op_cls: return None
            
            if not op_cls.is_valid(new_op_input_shapes):
                return None # Invalid for this shape
                
            # Compute new output shape
            try:
                new_out_shape = op_cls.get_output_shape(new_op_input_shapes, old_op.params)
            except:
                return None
                
            # Create new output tensor
            new_out = Tensor(f"{old_op.output.name}_gen", new_out_shape)
            old_to_new[old_op.output] = new_out
            
            new_op = Operator(old_op.op_type, new_op_inputs, new_out, old_op.params)
            new_ops.append(new_op)
            
        # 3. Outputs
        new_outputs = [old_to_new[t] for t in template_graph.outputs]
        
        return Graph(new_inputs, new_ops, new_outputs)

    def verify_generalization(self, g1: Graph, g2: Graph) -> bool:
        # Define test suites
        test_shapes = [
            {"A": (3, 3), "B": (3, 3)},
            {"A": (4, 4), "B": (4, 4)},
            {"A": (1, 5), "B": (1, 5)},
            {"A": (5, 1), "B": (5, 1)},
            {"A": (2, 3), "B": (3, 2)}, # MatMul friendly
            {"A": (3, 2), "B": (2, 3)},
            {"A": (10, 10), "B": (10, 10)},
            {"A": (1, 1), "B": (1, 1)},
        ]
        
        const_dict = self.get_constants_dict()
        const_names = set(const_dict.keys())
        
        required_inputs = set(t.name for t in g1.inputs)
        required_vars = required_inputs - const_names
        
        success_count = 0
        
        for shapes in test_shapes:
            # Check if this test shape config provides all required variables
            if not required_vars.issubset(shapes.keys()):
                continue
                
            # Prepare config for instantiation
            current_config = {}
            
            # Add vars
            for var_name in required_vars:
                current_config[var_name] = shapes[var_name]
                
            # Add constants
            for const_name in required_inputs.intersection(const_names):
                current_config[const_name] = ()
                
            # Instantiate
            g1_new = self.instantiate_graph(g1, current_config)
            g2_new = self.instantiate_graph(g2, current_config)
            
            if g1_new is None and g2_new is None:
                continue 
            
            if g1_new is None or g2_new is None:
                return False
                
            # Both valid, check values
            inputs = {}
            for name in required_vars:
                inputs[name] = np.random.normal(0, 1, current_config[name]).astype(np.float32)
            
            inputs.update(const_dict)
                
            try:
                res1 = self.evaluator.evaluate(g1_new, inputs)
                res2 = self.evaluator.evaluate(g2_new, inputs)
                
                val1 = res1[g1_new.outputs[0].name]
                val2 = res2[g2_new.outputs[0].name]
                
                if not np.allclose(val1, val2, rtol=1e-5, atol=self.epsilon):
                    return False
                
                success_count += 1
            except:
                return False
                
        return success_count > 0

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
            
            # Add to fingerprint map
            fp = self.get_fingerprint(input_data[name])
            if fp not in self.fingerprint_map:
                self.fingerprint_map[fp] = []
            self.fingerprint_map[fp].append(t)
            
        # 1.5 Initialize Constants (0, 1, -1)
        # We create scalar tensors that broadcast to input shapes or just exist as scalars
        # For simplicity, let's add scalar tensors. 
        # Note: Ops need to support broadcasting if we mix scalars and tensors.
        # Most numpy ops support broadcasting.
        
        for c in self.constants:
            name = f"Const({c})"
            # Scalar shape is empty tuple
            shape = () 
            t = Tensor(name, shape)
            self.tensors.append(t)
            self.tensor_values[t] = np.array(c, dtype=np.float32)
            
            fp = self.get_fingerprint(self.tensor_values[t])
            if fp not in self.fingerprint_map:
                self.fingerprint_map[fp] = []
            self.fingerprint_map[fp].append(t)
        
        # 2. Enumeration Loop
        print(f"Enumerating up to {self.max_ops} operators...")
        
        # We iterate by "depth" effectively by processing the list of tensors
        # But since we append to self.tensors, we need to be careful not to infinite loop or re-process too much
        # A simple approach: generations.
                
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
        
        # Calculate fingerprint
        fp = self.get_fingerprint(out_val)
        
        # Check for equivalence using fingerprint map
        candidates = self.fingerprint_map.get(fp, [])
        
        is_new = True
        for existing_t in candidates:
            if existing_t.shape == out_shape:
                if self.check_equivalence(existing_t, out_val): # We need to temporarily store value for out_tensor
                    # Found a rule!
                    # existing_t is equivalent to out_tensor
                    # But we only want to record it if the graphs are structurally different
                    # And we don't want to add out_tensor to the pool if it's redundant (pruning)
                    
                    # Construct graphs
                    g1 = self.reconstruct_graph(existing_t)
                    
                    # Actually, reconstruct_graph needs to work on the new op too
                    # So we temporarily register it
                    self.tensor_to_creator[out_tensor] = op
                    g2_full = self.reconstruct_graph(out_tensor)
                    
                    # Verify with more random inputs
                    if self.verify_equivalence(g1, g2_full):
                        # Structural check using canonical signature
                        if g1.get_structural_signature() != g2_full.get_structural_signature():
                            # Check generalization
                            is_general = self.verify_generalization(g1, g2_full)
                            
                            # We want simplification rules: Complex -> Simple
                            # g1 is existing (simpler/older), g2_full is new (complex/newer)
                            self.rules.append({
                                'source': g2_full,
                                'target': g1,
                                'is_general': is_general
                            })
                            # print(f"Found Rule: {g2_full}  ==>  {g1}")
                    
                    is_new = False
                    del self.tensor_to_creator[out_tensor] # Cleanup
                    break
        
        if is_new:
            # Add to list
            self.tensor_to_creator[out_tensor] = op
            self.tensor_values[out_tensor] = out_val
            new_tensors_list.append(out_tensor)
            
            # Add to fingerprint map
            if fp not in self.fingerprint_map:
                self.fingerprint_map[fp] = []
            self.fingerprint_map[fp].append(out_tensor)



    
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
        for i, rule in enumerate(self.rules):
            # Handle both old tuple format (if any) and new dict format
            if isinstance(rule, tuple):
                g1, g2 = rule
                is_general = False
            else:
                g1 = rule['target'] # Simpler
                g2 = rule['source'] # Complex
                is_general = rule.get('is_general', False)
                
            data.append({
                "id": i + 1,
                "source": serialize_graph(g2),
                "target": serialize_graph(g1),
                "is_general": is_general
            })
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
