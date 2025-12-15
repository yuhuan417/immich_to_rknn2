#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def batch_convert_models():
    cumsum_dir = Path("cumsum")
    target_platform = "rk3566"
    model_list_file = Path("model_list")
    
    if not cumsum_dir.exists():
        print(f"Error: {cumsum_dir} directory not found")
        return 1
    
    if not model_list_file.exists():
        print(f"Error: {model_list_file} not found")
        return 1
    
    with open(model_list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip()]
    
    total = len(model_paths)
    success_count = 0
    failed_models = []
    
    for idx, model_path in enumerate(model_paths, 1):
        model_name = model_path.split('/')[0]
        model_type = model_path.split('/')[1] if '/' in model_path else None
        
        source_model_dir = cumsum_dir / model_name
        
        if not source_model_dir.exists():
            print(f"\n[{idx}/{total}] Skipping {model_name}: not found in cumsum directory")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{total}] Converting {model_name}")
        print(f"{'='*80}")
        
        import shutil
        
        try:
            cmd = [
                sys.executable,
                "build_rknn.py",
                str(source_model_dir),
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
            
            print(f"✓ Successfully converted {model_name}")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to convert {model_name}")
            print(f"Error: {e}")
            failed_models.append(model_name)
        except Exception as e:
            print(f"✗ Unexpected error converting {model_name}: {e}")
            failed_models.append(model_name)
    
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
    sys.exit(batch_convert_models())
