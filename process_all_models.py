import subprocess
import os
import sys

def run_conversion(input_path, output_path):
    print(f"Running conversion: {input_path} -> {output_path}")
    
    # Check input existence
    if not os.path.exists(input_path):
        print(f"Skipping: Input {input_path} not found.")
        return

    cmd = [sys.executable, "replace_cumsum.py", "--input", input_path, "--output", output_path]
    
    try:
        # Run in a separate process
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"Success: {input_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")
        # We might want to continue processing other models even if one fails
        
def main():
    # 1. Base Model
    base_dir = "nllb-clip-base-siglip__v1"
    out_dir = "test_cumsum"
    
    tasks = [
        (os.path.join(base_dir, "textual/model.onnx"), os.path.join(out_dir, "textual/model.onnx")),
        (os.path.join(base_dir, "visual/model.onnx"), os.path.join(out_dir, "visual/model.onnx")),
    ]

    # 2. XLM Model
    xlm_dir = "XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k"
    xlm_out_dir = "test_cumsum_xlm" 
    
    tasks.append((os.path.join(xlm_dir, "textual/model.onnx"), os.path.join(xlm_out_dir, "textual/model.onnx")))
    tasks.append((os.path.join(xlm_dir, "visual/model.onnx"), os.path.join(xlm_out_dir, "visual/model.onnx")))

    for input_p, output_p in tasks:
        run_conversion(input_p, output_p)

if __name__ == "__main__":
    main()
