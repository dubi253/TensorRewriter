import numpy as np
from tensor_rewriter.ops import OPS_REGISTRY, MatMul, Reshape, Concat, Slice, Sum, ExpandDims

def test_matmul():
    op = MatMul
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]])]
    res = op.compute(inputs)
    assert np.array_equal(res, inputs[0])
    
    assert op.is_valid([(2, 2), (2, 2)])
    assert not op.is_valid([(2, 3), (2, 2)])

def test_reshape():
    op = Reshape
    inputs = [np.array([[1, 2], [3, 4]])]
    params = {'shape': (4,)}
    res = op.compute(inputs, params)
    assert res.shape == (4,)
    assert np.array_equal(res, [1, 2, 3, 4])
    
    valid_params = op.get_valid_params([(2, 2)])
    assert {'shape': (4,)} in valid_params
    assert {'shape': (1, 4)} in valid_params

def test_concat():
    op = Concat
    inputs = [np.array([[1, 2]]), np.array([[3, 4]])]
    params = {'axis': 0}
    res = op.compute(inputs, params)
    assert res.shape == (2, 2)
    assert np.array_equal(res, [[1, 2], [3, 4]])
    
    params = {'axis': 1}
    res = op.compute(inputs, params)
    assert res.shape == (1, 4)
    assert np.array_equal(res, [[1, 2, 3, 4]])

def test_slice():
    op = Slice
    inputs = [np.array([[1, 2], [3, 4]])]
    params = {'axis': 0, 'start': 0, 'end': 1}
    res = op.compute(inputs, params)
    assert res.shape == (1, 2)
    assert np.array_equal(res, [[1, 2]])
    
    params = {'axis': 1, 'start': 1, 'end': 2}
    res = op.compute(inputs, params)
    assert res.shape == (2, 1)
    assert np.array_equal(res, [[2], [4]])

def test_sum():
    op = Sum
    inputs = [np.array([[1, 2], [3, 4]])]
    params = {'axis': 0}
    res = op.compute(inputs, params)
    assert res.shape == (2,)
    assert np.array_equal(res, [4, 6])

def test_expand_dims():
    op = ExpandDims
    inputs = [np.array([1, 2])]
    params = {'axis': 0}
    res = op.compute(inputs, params)
    assert res.shape == (1, 2)
    assert np.array_equal(res, [[1, 2]])
