import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_rewriter.synthesizer import Synthesizer

def main():
    # Define initial inputs
    input_config = {
        "A": (2, 2),
        "B": (2, 2)
    }
    
    print("Starting synthesis with inputs:", input_config)
    synth = Synthesizer(input_config, max_ops=4)
    synth.synthesize()
    
    output_path = os.path.join(os.path.dirname(__file__), "../web/src/rules.json")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    synth.export_rules(output_path)
    print(f"Rules exported to {output_path}")

if __name__ == "__main__":
    main()
