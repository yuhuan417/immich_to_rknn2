#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import shutil

def batch_convert_to_rknn():
    cumsum_dir = Path("cumsum")
    rknn_result_dir = Path("rknn_result")
    model_list_file = Path("model_list")
    target_platform = "rk3566"
    
    if not cumsum_dir.exists():
        print(f"Error: {cumsum_dir} directory not found")
        return 1
    
    if not model_list_file.exists():
        print(f"Error: {model_list_file} not found")
        return 1
    
    rknn_result_dir.mkdir(exist_ok=True)
    
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
        print(f"[{idx}/{total}] Converting {model_path}")
        print(f"{'='*80}")
        
        source_onnx = cumsum_dir / model_path / "model.onnx"
        if not source_onnx.exists():
            print(f"Error: {source_onnx} not found")
            failed_models.append(model_path)
            continue
        
        output_dir = rknn_result_dir / model_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for item in (cumsum_dir / model_path).iterdir():
            dest = output_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
        
        print(f"Copied {cumsum_dir / model_path} -> {output_dir}")
        
        output_onnx = output_dir / "model.onnx"
        
        try:
            cmd = [
                sys.executable,
                "build_rknn.py",
                str(output_onnx),
                target_platform
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                check=True,
                capture_output=False,
                text=True
            )
            
            output_rknn = output_dir / f"{target_platform}.rknn"
            if output_rknn.exists():
                print(f"✓ Successfully converted {model_path}")
                success_count += 1
            else:
                print(f"✗ RKNN file not generated for {model_path}")
                failed_models.append(model_path)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to convert {model_path}")
            print(f"Error code: {e.returncode}")
            failed_models.append(model_path)
        except Exception as e:
            print(f"✗ Unexpected error converting {model_path}: {e}")
            import traceback
            traceback.print_exc()
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
    sys.exit(batch_convert_to_rknn())
