import onnx
import onnxruntime as ort
import numpy as np
import os

def verify_model(original_path, modified_path, sample_input_shape=(1, 77), n_iters=3):
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
    sess_opts.log_severity_level = 3
    
    # Load original model first, get metadata, then release it
    try:
        session_orig = ort.InferenceSession(original_path, sess_opts)
    except Exception as e:
        print(f"Warning: Could not create inference session for original model: {e}")
        return False
    
    # Get input metadata from original model
    input_metas = session_orig.get_inputs()
    output_names = [o.name for o in session_orig.get_outputs()]
    
    print(f"Inputs: {len(input_metas)}")
    for meta in input_metas:
        print(f"  - {meta.name}, Shape: {meta.shape}, Type: {meta.type}")
    print(f"Comparing Model Outputs: {output_names}")
    
    # Generate input data for all inputs
    def generate_input_data():
        input_feed = {}
        for input_meta in input_metas:
            input_name = input_meta.name
            input_shape = input_meta.shape
            input_type = input_meta.type
            
            # Build dummy shape
            dummy_shape = []
            for d in input_shape:
                if isinstance(d, str) or d is None:
                    dummy_shape.append(1)
                elif d <= 0:
                    dummy_shape.append(1)
                else:
                    dummy_shape.append(d)
            
            # Generate data based on type
            if "int" in str(input_type).lower():
                data = np.random.randint(0, 1000, size=dummy_shape, dtype=np.int32)
            else:
                data = np.random.randn(*dummy_shape).astype(np.float32)
            
            input_feed[input_name] = data
        
        return input_feed
    
    # Generate first input
    data = generate_input_data()
    
    try:
        out_orig_sample = session_orig.run(None, data)
        # Print output shapes
        for j, o in enumerate(out_orig_sample):
            print(f"  Output {j} ({output_names[j]}): Shape {o.shape}")
    except Exception as e:
        print(f"Runtime Check Failed on original model: {e}")
        session_orig = None  # Ensure cleanup
        return False
    
    # Release original model from memory
    session_orig = None
    
    # Now load modified model
    try:
        session_mod = ort.InferenceSession(modified_path, sess_opts)
    except Exception as e:
        print(f"Warning: Could not create inference session for modified model: {e}")
        return False
    
    # Run multiple iterations
    print(f"Running {n_iters} iterations with random inputs...")
    
    max_diff_all = 0.0
    all_matched = True
    
    for i in range(n_iters):
        # Generate input data for all inputs
        data = generate_input_data()

        # Run modified model
        try:
            out_mod = session_mod.run(None, data)
        except Exception as e:
            print(f"Runtime Check Failed iter {i} on modified model: {e}")
            session_mod = None  # Ensure cleanup
            return False
        
        # For comparison, we need to reload original model for each iteration
        try:
            session_orig = ort.InferenceSession(original_path, sess_opts)
            out_orig = session_orig.run(None, data)
            session_orig = None  # Release immediately after use
        except Exception as e:
            print(f"Runtime Check Failed iter {i} on original model: {e}")
            session_mod = None  # Ensure cleanup
            return False

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
        return True
    else:
        print(f"FAILED: Some outputs mismatched. Max Diff: {max_diff_all}")
        return False

def compare_models_from_list(list_file, models_dir="models", cumsum_dir="cumsum", n_iters=3):
    """从 model_list 文件读取模型列表并对比"""
    if not os.path.exists(list_file):
        print(f"Error: Model list file {list_file} does not exist.")
        return
    
    if not os.path.exists(models_dir) or not os.path.exists(cumsum_dir):
        print(f"Error: Directory {models_dir} or {cumsum_dir} does not exist.")
        return
    
    # 读取模型列表
    with open(list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(model_paths)} models in {list_file}")
    
    results = []
    for rel_path in model_paths:
        # 构造完整路径
        original_path = os.path.join(models_dir, rel_path, "model.onnx")
        modified_path = os.path.join(cumsum_dir, rel_path, "model.onnx")
        
        if not os.path.exists(original_path):
            print(f"\n⊗ Skipped: {rel_path} (original model not found)")
            continue
            
        if not os.path.exists(modified_path):
            print(f"\n⊗ Skipped: {rel_path} (modified model not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Comparing: {rel_path}")
        print(f"  Original: {original_path}")
        print(f"  Modified: {modified_path}")
        print('='*70)
        
        try:
            result = verify_model(original_path, modified_path, n_iters=n_iters)
            results.append((rel_path, result))
        except Exception as e:
            print(f"✗ Error comparing {rel_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append((rel_path, False))
    
    # 打印总结
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    success_count = sum(1 for _, r in results if r is not False)
    total_count = len(results)
    
    for model_path, result in results:
        status = "✓" if result is not False else "✗"
        print(f"{status} {model_path}")
    
    print(f"\nTotal: {success_count}/{total_count} models verified successfully")

def compare_models_in_directories(models_dir="models", cumsum_dir="cumsum", n_iters=3):
    """比较 models/ 和 cumsum/ 目录下的对应模型"""
    import glob
    
    if not os.path.exists(models_dir) or not os.path.exists(cumsum_dir):
        print(f"Error: Directory {models_dir} or {cumsum_dir} does not exist.")
        return
    
    # 查找所有模型
    model_patterns = [
        "*/textual/model.onnx",
        "*/visual/model.onnx"
    ]
    
    results = []
    for pattern in model_patterns:
        original_models = glob.glob(os.path.join(models_dir, pattern))
        
        for original_path in original_models:
            # 构造对应的 cumsum 路径
            rel_path = os.path.relpath(original_path, models_dir)
            modified_path = os.path.join(cumsum_dir, rel_path)
            
            if not os.path.exists(modified_path):
                print(f"\n⊗ Skipped: {rel_path} (modified model not found)")
                continue
            
            print(f"\n{'='*70}")
            print(f"Comparing: {rel_path}")
            print(f"  Original: {original_path}")
            print(f"  Modified: {modified_path}")
            print('='*70)
            
            try:
                result = verify_model(original_path, modified_path, n_iters=n_iters)
                results.append((rel_path, result))
            except Exception as e:
                print(f"✗ Error comparing {rel_path}: {e}")
                results.append((rel_path, False))
    
    # 打印总结
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    success_count = sum(1 for _, r in results if r is not False)
    total_count = len(results)
    
    for model_path, result in results:
        status = "✓" if result is not False else "✗"
        print(f"{status} {model_path}")
    
    print(f"\nTotal: {success_count}/{total_count} models verified successfully")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify ONNX model replacement correctness")
    parser.add_argument("--original", help="Path to original model")
    parser.add_argument("--modified", help="Path to modified model")
    parser.add_argument("--compare-dirs", action="store_true", 
                       help="Compare all models in models/ and cumsum/ directories")
    parser.add_argument("--model-list", help="Path to file containing list of models to compare")
    parser.add_argument("--models-dir", default="models", 
                       help="Directory containing original models (default: models)")
    parser.add_argument("--cumsum-dir", default="cumsum", 
                       help="Directory containing modified models (default: cumsum)")
    parser.add_argument("--n-iters", type=int, default=3,
                       help="Number of iterations for verification (default: 3)")
    
    args = parser.parse_args()
    
    if args.model_list:
        compare_models_from_list(args.model_list, args.models_dir, args.cumsum_dir, args.n_iters)
    elif args.compare_dirs:
        compare_models_in_directories(args.models_dir, args.cumsum_dir, args.n_iters)
    elif args.original and args.modified:
        if not os.path.exists(args.original):
            print(f"Error: Original model {args.original} does not exist.")
            return
        if not os.path.exists(args.modified):
            print(f"Error: Modified model {args.modified} does not exist.")
            return
        
        verify_model(args.original, args.modified, n_iters=args.n_iters)
    else:
        print("Error: Please specify either:")
        print("  1. --original and --modified for single model comparison")
        print("  2. --compare-dirs for batch comparison")
        print("  3. --model-list for comparing models from a list file")
        print("\nUse --help for more information.")

if __name__ == "__main__":
    main()
