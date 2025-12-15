#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import shutil
from replace_cumsum import process_model

def batch_replace_cumsum():
    models_dir = Path("models")
    cumsum_dir = Path("cumsum")
    model_list_file = Path("model_list")
    
    if not models_dir.exists():
        print(f"Error: {models_dir} directory not found")
        return 1
    
    if not model_list_file.exists():
        print(f"Error: {model_list_file} not found")
        return 1
    
    cumsum_dir.mkdir(exist_ok=True)
    
    with open(model_list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip()]
    
    total = len(model_paths)
    success_count = 0
    failed_models = []
    
    for idx, model_path in enumerate(model_paths, 1):
        parts = model_path.split('/')
        model_name = parts[0]
        model_type = parts[1] if len(parts) > 1 else None
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{total}] Processing {model_path}")
        print(f"{'='*80}")
        
        source_model_dir = models_dir / model_name
        if not source_model_dir.exists():
            print(f"Error: {source_model_dir} not found")
            failed_models.append(model_path)
            continue
        
        output_model_dir = cumsum_dir / model_name
        output_model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type == "textual":
            source_textual_dir = source_model_dir / "textual"
            source_onnx = source_textual_dir / "model.onnx"
            
            if not source_onnx.exists():
                print(f"Error: {source_onnx} not found")
                failed_models.append(model_path)
                continue
            
            output_textual_dir = output_model_dir / "textual"
            output_textual_dir.mkdir(parents=True, exist_ok=True)
            output_onnx = output_textual_dir / "model.onnx"
            
            try:
                print(f"Converting {source_onnx} -> {output_onnx}")
                process_model(str(source_onnx), str(output_onnx))
                
                if (source_textual_dir / "model.onnx.data").exists():
                    print(f"Note: Original model has external data file")
                
                if (source_model_dir / "visual").exists():
                    print(f"Copying visual directory...")
                    output_visual_dir = output_model_dir / "visual"
                    if output_visual_dir.exists():
                        shutil.rmtree(output_visual_dir)
                    shutil.copytree(source_model_dir / "visual", output_visual_dir)
                
                print(f"✓ Successfully converted {model_path}")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Failed to convert {model_path}: {e}")
                import traceback
                traceback.print_exc()
                failed_models.append(model_path)
        
        elif model_type == "visual":
            source_visual_dir = source_model_dir / "visual"
            source_onnx = source_visual_dir / "model.onnx"
            
            if not source_onnx.exists():
                print(f"Error: {source_onnx} not found")
                failed_models.append(model_path)
                continue
            
            output_visual_dir = output_model_dir / "visual"
            output_visual_dir.mkdir(parents=True, exist_ok=True)
            output_onnx = output_visual_dir / "model.onnx"
            
            try:
                print(f"Converting {source_onnx} -> {output_onnx}")
                process_model(str(source_onnx), str(output_onnx))
                
                if (source_visual_dir / "model.onnx.data").exists():
                    print(f"Note: Original model has external data file")
                
                print(f"✓ Successfully converted {model_path}")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Failed to convert {model_path}: {e}")
                import traceback
                traceback.print_exc()
                failed_models.append(model_path)
        else:
            print(f"Warning: Unknown model type for {model_path}")
            failed_models.append(model_path)
    
    print(f"\n{'='*80}")
    print(f"Conversion Summary")
    print(f"{'='*80}")
    print(f"Total models: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_models)}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(batch_replace_cumsum())
