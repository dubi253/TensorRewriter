import numpy as np
from typing import List, Tuple, Dict, Any
import itertools

OPS_REGISTRY = {}

def register_op(name):
    def decorator(cls):
        OPS_REGISTRY[name] = cls
        return cls
    return decorator

class OpImplementation:
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        raise NotImplementedError
    
    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return True
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{}]

@register_op("MatMul")
class MatMul(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.matmul(inputs[0], inputs[1])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        # Handle broadcasting for MatMul? 
        # Standard MatMul: (..., m, k) x (..., k, n) -> (..., m, n)
        # For now, stick to simple 2D or compatible ranks
        s1 = input_shapes[0]
        s2 = input_shapes[1]
        if len(s1) < 2 or len(s2) < 2:
             # Fallback for scalars/vectors if numpy allows, but MatMul usually implies matrices
             # If one is scalar, it's not MatMul, it's Mul.
             raise ValueError("MatMul requires at least 2D inputs")
             
        return s1[:-1] + (s2[-1],)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 2: return False
        s1 = input_shapes[0]
        s2 = input_shapes[1]
        if len(s1) < 2 or len(s2) < 2: return False
        return s1[-1] == s2[-2]

@register_op("ElementWiseAdd")
class ElementWiseAdd(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.add(inputs[0], inputs[1])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        # Broadcasting logic
        s1 = input_shapes[0]
        s2 = input_shapes[1]
        if s1 == s2: return s1
        if not s1: return s2 # s1 is scalar
        if not s2: return s1 # s2 is scalar
        # Simple broadcast check: if one is scalar-like or same shape
        # Full numpy broadcast is complex to implement manually, 
        # but for synthesis we can rely on numpy's behavior during compute 
        # and just return the larger shape if valid.
        # For is_valid, we need to be stricter.
        return np.broadcast_shapes(s1, s2)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 2: return False
        try:
            np.broadcast_shapes(input_shapes[0], input_shapes[1])
            return True
        except:
            return False

@register_op("ElementWiseSub")
class ElementWiseSub(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.subtract(inputs[0], inputs[1])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        return np.broadcast_shapes(input_shapes[0], input_shapes[1])

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 2: return False
        try:
            np.broadcast_shapes(input_shapes[0], input_shapes[1])
            return True
        except:
            return False


@register_op("Transpose")
class Transpose(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.transpose(inputs[0])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        return tuple(reversed(input_shapes[0]))

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 1: return False
        return True

@register_op("Relu")
class Relu(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.maximum(inputs[0], 0)
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        return input_shapes[0]

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 1: return False
        return True

@register_op("Reshape")
class Reshape(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.reshape(inputs[0], params['shape'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        return params['shape']

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 1: return False
        return True
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        # Generate some valid reshapes
        shape = input_shapes[0]
        num_elements = np.prod(shape)
        
        valid_shapes = []
        # 1D
        valid_shapes.append((int(num_elements),))
        
        # 2D factors
        for i in range(1, int(np.sqrt(num_elements)) + 1):
            if num_elements % i == 0:
                j = num_elements // i
                valid_shapes.append((i, j))
                if i != j:
                    valid_shapes.append((j, i))
        
        # 3D factors (simple heuristic: split one dim of 2D shapes)
        # This is a basic expansion to support 3D reshapes
        for s2 in list(valid_shapes):
            if len(s2) == 2:
                d1, d2 = s2
                # Split d1
                for k in range(1, int(np.sqrt(d1)) + 1):
                    if d1 % k == 0:
                        valid_shapes.append((k, d1//k, d2))
                # Split d2
                for k in range(1, int(np.sqrt(d2)) + 1):
                    if d2 % k == 0:
                        valid_shapes.append((d1, k, d2//k))
                    
        # Filter out the original shape to avoid identity (though identity is a valid rule)
        # Actually, identity rules like Reshape(A, shape=A.shape) = A are good to find.
        
        # Deduplicate
        unique_shapes = sorted(list(set(valid_shapes)))
        
        return [{'shape': s} for s in unique_shapes]

@register_op("Concat")
class Concat(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.concatenate(inputs, axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        axis = params['axis']
        new_dim = sum(s[axis] for s in input_shapes)
        lst = list(input_shapes[0])
        lst[axis] = new_dim
        return tuple(lst)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) < 2: return False
        # All shapes must match except on axis. But we don't know axis yet.
        # We check if there EXISTS an axis.
        # Actually, get_valid_params will determine valid axes.
        # Here we just check if ranks match.
        rank = len(input_shapes[0])
        for s in input_shapes[1:]:
            if len(s) != rank: return False
        return True
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        rank = len(input_shapes[0])
        valid_axes = []
        for axis in range(rank):
            # Check if all other dims match
            match = True
            base_shape = list(input_shapes[0])
            base_shape[axis] = -1
            for s in input_shapes[1:]:
                check_shape = list(s)
                check_shape[axis] = -1
                if base_shape != check_shape:
                    match = False
                    break
            if match:
                valid_axes.append(axis)
        return [{'axis': a} for a in valid_axes]

@register_op("Slice")
class Slice(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        inp = inputs[0]
        axis = params['axis']
        start = params['start']
        end = params['end']
        # Construct slice object
        slices = [slice(None)] * inp.ndim
        slices[axis] = slice(start, end)
        return inp[tuple(slices)]
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        axis = params['axis']
        start = params['start']
        end = params['end']
        shape[axis] = end - start
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        shape = input_shapes[0]
        params_list = []
        for axis, dim in enumerate(shape):
            for start in range(dim):
                for end in range(start + 1, dim + 1):
                    params_list.append({'axis': axis, 'start': start, 'end': end})
        return params_list

@register_op("Sum")
class Sum(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.sum(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        del shape[params['axis']]
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]))]

@register_op("ExpandDims")
class ExpandDims(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.expand_dims(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        shape.insert(params['axis'], 1)
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]) + 1)]

@register_op("ElementWiseMul")
class ElementWiseMul(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.multiply(inputs[0], inputs[1])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        return np.broadcast_shapes(input_shapes[0], input_shapes[1])

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        if len(input_shapes) != 2: return False
        try:
            np.broadcast_shapes(input_shapes[0], input_shapes[1])
            return True
        except:
            return False

@register_op("Mean")
class Mean(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.mean(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        del shape[params['axis']]
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]))]

@register_op("Max")
class Max(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.max(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        del shape[params['axis']]
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]))]

@register_op("Min")
class Min(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.min(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        del shape[params['axis']]
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]))]

@register_op("Prod")
class Prod(OpImplementation):
    @staticmethod
    def compute(inputs: List[np.ndarray], params: Dict[str, Any] = None) -> np.ndarray:
        return np.prod(inputs[0], axis=params['axis'])
    
    @staticmethod
    def get_output_shape(input_shapes: List[Tuple[int, ...]], params: Dict[str, Any] = None) -> Tuple[int, ...]:
        shape = list(input_shapes[0])
        del shape[params['axis']]
        return tuple(shape)

    @staticmethod
    def is_valid(input_shapes: List[Tuple[int, ...]]) -> bool:
        return len(input_shapes) == 1
    
    @staticmethod
    def get_valid_params(input_shapes: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        return [{'axis': i} for i in range(len(input_shapes[0]))]

