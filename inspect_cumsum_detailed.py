import onnx
import sys
from onnx import numpy_helper

def inspect(path):
    print(f"\nInspecting {path}")
    try:
        model = onnx.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return

    graph = model.graph
    
    # Map initializers
    initializers = {init.name: init for init in graph.initializer}
    
    print("Graph Inputs:")
    for inp in graph.input:
        dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")
    
    for node in graph.node:
        if node.op_type == 'CumSum':
            print(f"Node: {node.name}")
            
            # Print inputs
            for i, input_name in enumerate(node.input):
                print(f"  Input {i}: {input_name}")
                if input_name in initializers:
                    val = numpy_helper.to_array(initializers[input_name])
                    print(f"    Value (Initializer): {val}")
                else:
                    # Find producer
                    producer = next((n for n in graph.node if input_name in n.output), None)
                    if producer:
                        print(f"    Producer: {producer.name} ({producer.op_type})")
                        if producer.op_type == 'Constant':
                             for attr in producer.attribute:
                                 if attr.name == 'value':
                                     val = numpy_helper.to_array(attr.t)
                                     print(f"    Value (Constant): {val}")
                    else:
                        print(f"    Producer: Graph Input?")
            
            # Print attributes
            for attr in node.attribute:
                print(f"  Attr: {attr.name} = {attr}")
                
            # Check shape of input 0
            input_name = node.input[0]
            print(f"  Shape of {input_name}:")
            found_shape = False
            for val_info in graph.value_info:
                if val_info.name == input_name:
                    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in val_info.type.tensor_type.shape.dim]
                    print(f"    Inferred (ValueInfo): {dims}")
                    found_shape = True
                    break
            if not found_shape:
                 # Check inputs
                 for inp in graph.input:
                     if inp.name == input_name:
                         dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in inp.type.tensor_type.shape.dim]
                         print(f"    Graph Input: {dims}")
                         found_shape = True
                         break
            if not found_shape:
                print("    Shape not found in ValueInfo or Graph Input")
            
            # Trace input 0
            curr_node = node
            for depth in range(3):
                input_name = curr_node.input[0]
                producer = next((n for n in graph.node if input_name in n.output), None)
                if producer:
                    print(f"    [Depth {depth+1}] Producer of {input_name}: {producer.name} ({producer.op_type})")
                    curr_node = producer
                    # If constant, print inputs/values
                    for i, inp in enumerate(producer.input):
                        if inp in initializers:
                             # Don't print huge arrays
                             arr = numpy_helper.to_array(initializers[inp])
                             print(f"      Input {i} (Init): Shape {arr.shape}, Type {arr.dtype}")
                        else:
                             print(f"      Input {i}: {inp}")
                else:
                    print(f"    [Depth {depth+1}] Producer is Graph Input or Missing")
                    break

if __name__ == "__main__":
    print("--- NLLB ---")
    inspect("nllb-clip-base-siglip__v1/textual/model.onnx")
    print("\n--- XLM ---")
    inspect("XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k/textual/model.onnx")
