import onnx
import sys
import os

def check(path):
    if not os.path.exists(path):
        print(f"{path}: File not found")
        return
        
    try:
        model = onnx.load(path)
        nodes = [n for n in model.graph.node if n.op_type == 'CumSum']
        print(f"{path}: Found {len(nodes)} CumSum nodes")
        for n in nodes:
            print(f"  -Name: {n.name}")
            print(f"   Inputs: {n.input}")
            print(f"   Outputs: {n.output}")
            for attr in n.attribute:
                print(f"   Attr: {attr.name} = {attr.i}")
    except Exception as e:
        print(f"{path}: Error loading - {e}")

print("--- Checking XLM-Roberta ---")
check("XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k/textual/model.onnx")
check("XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k/visual/model.onnx")
print("--- Checking NLLB ---")
check("nllb-clip-base-siglip__v1/textual/model.onnx")
check("nllb-clip-base-siglip__v1/visual/model.onnx")
