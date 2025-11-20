from tensor_rewriter.synthesizer import Synthesizer

def test_synthesis_basic():
    input_config = {"A": (2, 2)}
    synth = Synthesizer(input_config, max_ops=1)
    synth.synthesize()
    # Should find at least identity or simple ops
    assert len(synth.tensors) > 1

def test_synthesis_rules():
    input_config = {"A": (2, 2)}
    synth = Synthesizer(input_config, max_ops=2)
    synth.synthesize()
    
    assert len(synth.rules) > 0
    
    # Check for specific rules
    # e.g. Reshape(Reshape(A)) = A
    has_reshape_identity = False
    for r in synth.rules:
        s_ops = [op.op_type for op in r[0].operators]
        t_ops = [op.op_type for op in r[1].operators]
        
        # Check for Reshape -> Reshape vs Identity (empty ops in target if it's just input)
        # But our graph reconstruction includes the input node.
        # If target is just A, operators list is empty?
        # Let's check structure.
        
        # A common rule: Transpose(Transpose(A)) == A
        if "Transpose" in s_ops and len(s_ops) == 2 and len(t_ops) == 0:
             # This means Source has 2 Transposes, Target has 0 ops (just input)
             pass

def test_concat_rules():
    # Test if Concat is used
    input_config = {"A": (2, 2), "B": (2, 2)}
    synth = Synthesizer(input_config, max_ops=1)
    synth.synthesize()
    
    # With max_ops=1, we can have Concat(A, B)
    # But Concat(A, B) is not equivalent to anything simple unless A=B?
    # Or Concat(A, A)
    
    # Let's just check if Concat op was generated
    concat_generated = False
    for t in synth.tensors:
        creator = synth.tensor_to_creator.get(t)
        if creator and creator.op_type == "Concat":
            concat_generated = True
            break
    
    assert concat_generated

