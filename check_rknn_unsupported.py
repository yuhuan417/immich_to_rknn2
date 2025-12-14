#!/usr/bin/env python3
"""
扫描 models 目录中的所有 ONNX 模型，检查其中是否包含 RKNN 不支持的算子。
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import onnx

# RKNN Toolkit2 不支持或有限制的 ONNX 算子列表
# 参考: https://huggingface.co/csukuangfj/rknn-toolkit2-v2.1.0-2024-08-08/blob/main/doc/RKNNToolKit2_OP_Support-v2.1.0.md
# RKNN Toolkit2 版本: v2.1.0, ONNX opset version: 19

RKNN_UNSUPPORTED_OPS = {
    # 明确标记为 "Not Supported" 的算子
    "Abs",                      # Not Supported
    "Acos",                     # Not Supported
    "Acosh",                    # Not Supported
    "Asin",                     # Not Supported
    "Asinh",                    # Not Supported
    "Atan",                     # Not Supported
    "Atanh",                    # Not Supported
    "Bernoulli",                # Not Supported
    "BitShift",                 # Not Supported
    "BitwiseAnd",               # Not Supported
    "BitwiseNot",               # Not Supported
    "BitwiseOr",                # Not Supported
    "BitwiseXor",               # Not Supported
    "BlackmanWindow",           # Not Supported
    "CastLike",                 # Not Supported
    "Ceil",                     # Not Supported
    "Celu",                     # Not Supported
    "CenterCropPad",            # Not Supported
    "Col2Im",                   # Not Supported
    "Compress",                 # Not Supported
    "ConcatFromSequence",       # Not Supported
    "ConvInteger",              # Not Supported
    "Cosh",                     # Not Supported
    "CumSum",                   # Not Supported
    "DeformConv",               # Not Supported
    "Det",                      # Not Supported
    "DFT",                      # Not Supported
    "DynamicQuantizeLinear",    # Not Supported
    "Einsum",                   # Not Supported
    "GlobalLpPool",             # Not Supported
    "GridSample",               # Not Supported
    "GroupNormalization",       # Not Supported
    "HammingWindow",            # Not Supported
    "HannWindow",               # Not Supported
    "Hardmax",                  # Not Supported
    "IsInf",                    # Not Supported
    "IsNaN",                    # Not Supported
    "Loop",                     # Not Supported
    "LpPool",                   # Not Supported
    "MatMulInteger",            # Not Supported
    "Mean",                     # Not Supported
    "MelWeightMatrix",          # Not Supported
    "Multinomial",              # Not Supported
    "Neg",                      # Not Supported
    "NegativeLogLikelihoodLoss",# Not Supported
    "NonMaxSuppression",        # Not Supported
    "NonZero",                  # Not Supported
    "Not",                      # Not Supported
    "OneHot",                   # Not Supported
    "Optional",                 # Not Supported
    "OptionalGetElement",       # Not Supported
    "OptionalHasElement",       # Not Supported
    "Or",                       # Not Supported
    "QLinearConv",              # Not Supported
    "QLinearMatMul",            # Not Supported
    "RandomNormal",             # Not Supported
    "RandomNormalLike",         # Not Supported
    "RandomUniform",            # Not Supported
    "RandomUniformLike",        # Not Supported
    "Range",                    # Not Supported
    "Reciprocal",               # Not Supported
    "ReduceL1",                 # Not Supported
    "ReduceL2",                 # Not Supported
    "ReduceLogSum",             # Not Supported
    "ReduceLogSumExp",          # Not Supported
    "ReduceProd",               # Not Supported
    "ReduceSumSquare",          # Not Supported
    "RNN",                      # Not Supported
    "Round",                    # Not Supported
    "Scan",                     # Not Supported
    "ScatterElements",          # Not Supported
    "Selu",                     # Not Supported
    "SequenceAt",               # Not Supported
    "SequenceConstruct",        # Not Supported
    "SequenceEmpty",            # Not Supported
    "SequenceErase",            # Not Supported
    "SequenceInsert",           # Not Supported
    "SequenceLength",           # Not Supported
    "SequenceMap",              # Not Supported
    "Shrink",                   # Not Supported
    "Sign",                     # Not Supported
    "Sinh",                     # Not Supported
    "Softsign",                 # Not Supported
    "SplitToSequence",          # Not Supported
    "STFT",                     # Not Supported
    "StringNormalizer",         # Not Supported
    "Sum",                      # Not Supported
    "Tan",                      # Not Supported
    "TfIdfVectorizer",          # Not Supported
    "ThresholdedRelu",          # Not Supported
    "TopK",                     # Not Supported
    "Trilu",                    # Not Supported
    "Unique",                   # Not Supported
    "Xor",                      # Not Supported
}

# 需要特别关注的算子 (有条件支持，需检查具体使用方式)
RKNN_PARTIAL_SUPPORT_OPS = {
    "EyeLike",          # only support constant input
    "GatherND",         # Not Supported (文档标记)
    "GRU",              # batchsize: 1
    "If",               # only support constant input
    "LogSoftmax",       # batchsize: 1
    "Resize",           # mode: nearest2d/bilinear only
    "RoiAlign",         # pool type: average, batchsize: 1
    "Slice",            # batchsize: 1
    "Softmax",          # batchsize: 1
    "Tile",             # batchsize: 1, not support broadcast
}


def get_all_ops(model_path: str) -> set:
    """获取模型中所有算子类型"""
    try:
        model = onnx.load(model_path)
        ops = set()
        for node in model.graph.node:
            ops.add(node.op_type)
        return ops
    except Exception as e:
        print(f"  错误: 无法加载模型 {model_path}: {e}", file=sys.stderr)
        return set()


def check_model(model_path: str) -> dict:
    """检查单个模型中的不支持算子"""
    ops = get_all_ops(model_path)
    unsupported = ops & RKNN_UNSUPPORTED_OPS
    partial = ops & RKNN_PARTIAL_SUPPORT_OPS
    return {
        "all_ops": ops,
        "unsupported": unsupported,
        "partial_support": partial,
    }


def scan_models_dir(models_dir: str) -> dict:
    """扫描 models 目录中的所有模型"""
    results = {}
    models_path = Path(models_dir)
    
    # 查找所有 model.onnx 文件
    onnx_files = list(models_path.rglob("model.onnx"))
    onnx_files.sort()
    
    print(f"找到 {len(onnx_files)} 个 ONNX 模型文件\n")
    
    for onnx_file in onnx_files:
        rel_path = onnx_file.relative_to(models_path)
        model_name = str(rel_path.parent)  # 例如: "ViT-B-16__openai/textual"
        
        print(f"检查: {model_name}")
        result = check_model(str(onnx_file))
        results[model_name] = result
        
        if result["unsupported"]:
            print(f"  ❌ 不支持的算子: {', '.join(sorted(result['unsupported']))}")
        if result["partial_support"]:
            print(f"  ⚠️  部分支持的算子: {', '.join(sorted(result['partial_support']))}")
        if not result["unsupported"] and not result["partial_support"]:
            print(f"  ✅ 所有算子均受支持")
    
    return results


def print_summary(results: dict):
    """打印汇总报告"""
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)
    
    # 统计不支持算子的出现频率
    unsupported_count = defaultdict(list)
    partial_count = defaultdict(list)
    models_with_issues = []
    models_ok = []
    
    for model_name, result in results.items():
        has_issue = False
        for op in result["unsupported"]:
            unsupported_count[op].append(model_name)
            has_issue = True
        for op in result["partial_support"]:
            partial_count[op].append(model_name)
        
        if has_issue:
            models_with_issues.append(model_name)
        else:
            models_ok.append(model_name)
    
    print(f"\n总模型数: {len(results)}")
    print(f"有问题的模型: {len(models_with_issues)}")
    print(f"正常模型: {len(models_ok)}")
    
    if unsupported_count:
        print(f"\n不支持的算子统计:")
        print("-" * 40)
        for op, models in sorted(unsupported_count.items(), key=lambda x: -len(x[1])):
            print(f"  {op}: 出现在 {len(models)} 个模型中")
            for m in models:  # 只显示前5个
                print(f"    - {m}")
    
    if partial_count:
        print(f"\n部分支持的算子统计:")
        print("-" * 40)
        for op, models in sorted(partial_count.items(), key=lambda x: -len(x[1])):
            print(f"  {op}: 出现在 {len(models)} 个模型中")
    
    if models_with_issues:
        print(f"\n需要处理的模型列表:")
        print("-" * 40)
        for m in sorted(models_with_issues):
            print(f"  - {m}")


def main():
    # 默认扫描 models 目录
    models_dir = "models"
    
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    
    if not os.path.isdir(models_dir):
        print(f"错误: 目录不存在: {models_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"扫描目录: {models_dir}")
    print("=" * 80)
    
    results = scan_models_dir(models_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
