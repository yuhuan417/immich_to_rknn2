import onnx
import onnxruntime as ort
import numpy as np
import os

def verify_model(original_path, modified_path, sample_input_shape=(1, 77)):
    print(f"\nVerifying {modified_path}...")
    
    # 1. Check Nodes
    model = onnx.load(modified_path)
    cumsum_nodes = [n for n in model.graph.node if n.op_type == 'CumSum']
    if len(cumsum_nodes) > 0:
        print(f"FAILED: Found {len(cumsum_nodes)} CumSum nodes in modified model!")
        return False
    else:
        print("SUCCESS: No CumSum nodes found.")

    # 2. Compare Outputs
    # We need to know input names and types
    sess_opts = ort.SessionOptions()
    # sess_opts.log_severity_level = 3
    
    # Load original model first, get metadata, then release it
    try:
        session_orig = ort.InferenceSession(original_path, sess_opts)
    except Exception as e:
        print(f"Warning: Could not create inference session for original model: {e}")
        return
    
    # Get input metadata from original model
    input_meta = session_orig.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_type = input_meta.type
    output_names = [o.name for o in session_orig.get_outputs()]
    
    print(f"Input: {input_name}, Shape: {input_shape}, Type: {input_type}")
    print(f"Comparing Model Outputs: {output_names}")
    
    # Run original model for first iteration to get output shapes
    dummy_shape = []
    for d in input_shape:
        if isinstance(d, str) or d is None:
            dummy_shape.append(1) # Default to 1
        elif d <= 0:
            dummy_shape.append(1)
        else:
            dummy_shape.append(d)
            
    # Generate first input
    if "int" in str(input_type).lower():
        data = np.random.randint(0, 1000, size=dummy_shape, dtype=np.int32) 
    else:
        data = np.random.randn(*dummy_shape).astype(np.float32)
    
    try:
        out_orig_sample = session_orig.run(None, {input_name: data})
        # Print output shapes
        for j, o in enumerate(out_orig_sample):
            print(f"  Output {j} ({output_names[j]}): Shape {o.shape}")
    except Exception as e:
        print(f"Runtime Check Failed on original model: {e}")
        session_orig = None  # Ensure cleanup
        return
    
    # Release original model from memory
    session_orig = None
    
    # Now load modified model
    try:
        session_mod = ort.InferenceSession(modified_path, sess_opts)
    except Exception as e:
        print(f"Warning: Could not create inference session for modified model: {e}")
        return
    
    # Run multiple iterations
    n_iters = 10
    print(f"Running {n_iters} iterations with random inputs...")
    
    max_diff_all = 0.0
    all_matched = True
    
    for i in range(n_iters):
        # Generate dummy input
        dummy_shape = []
        for d in input_shape:
            if isinstance(d, str) or d is None:
                dummy_shape.append(1) # Default to 1
            elif d <= 0:
                dummy_shape.append(1)
            else:
                dummy_shape.append(d)
                
        # If explicitly passed sample usage
        if "int" in str(input_type).lower():
            # Int64 usually
            data = np.random.randint(0, 1000, size=dummy_shape, dtype=np.int32) 
        else:
            data = np.random.randn(*dummy_shape).astype(np.float32)

        # Run modified model
        try:
            out_mod = session_mod.run(None, {input_name: data})
        except Exception as e:
            print(f"Runtime Check Failed iter {i} on modified model: {e}")
            session_mod = None  # Ensure cleanup
            return
        
        # For comparison, we need to reload original model for each iteration
        try:
            session_orig = ort.InferenceSession(original_path, sess_opts)
            out_orig = session_orig.run(None, {input_name: data})
            session_orig = None  # Release immediately after use
        except Exception as e:
            print(f"Runtime Check Failed iter {i} on original model: {e}")
            session_mod = None  # Ensure cleanup
            return

        # Compare
        for j, (o1, o2) in enumerate(zip(out_orig, out_mod)):
            diff = np.abs(o1 - o2).max()
            max_diff_all = max(max_diff_all, diff)
            if not np.allclose(o1, o2, atol=1e-4):
                output_name = output_names[j]
                print(f"Iter {i}: Output '{output_name}' Mismatch! Diff: {diff}")
                all_matched = False
            
    # Release modified model
    session_mod = None
    
    if all_matched:
        print(f"SUCCESS: All {n_iters} iterations matched. Max Diff: {max_diff_all}")
    else:
        print(f"FAILED: Some outputs mismatched. Max Diff: {max_diff_all}")

def main():
    # base_dir = "nllb-clip-base-siglip__v1"
    # out_dir = "test_cumsum"
    
    # print("--- Textual Model ---")
    # verify_model(
    #    os.path.join(base_dir, "textual/model.onnx"),
    #    os.path.join(out_dir, "textual/model.onnx")
    # )
    
    # Visual model didn't change, but good to check if it's identical/loadable
    # print("\n--- Visual Model ---")
    # verify_model(
    # 2. XLM
    xlm_dir = "XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k"
    xlm_out_dir = "test_cumsum_xlm"
    print("\n--- XLM Textual Model ---")
    text_in = os.path.join(xlm_dir, "textual/model.onnx")
    text_out = os.path.join(xlm_out_dir, "textual/model.onnx")
    if os.path.exists(text_out):
        verify_model(text_in, text_out)
    else:
        print("XLM model output not found.")

if __name__ == "__main__":
    main()
