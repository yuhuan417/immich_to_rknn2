# Immich to RKNN2 转换项目

## 项目概述

本项目专注于将 Immich 机器学习模型从 ONNX 格式转换为 RKNN 格式，以便在 Rockchip NPU 上高效运行。项目主要解决了 RKNN 工具链不支持某些 ONNX 操作（特别是 CumSum 操作）的问题。

## 核心功能

### 1. 模型转换
- **ONNX 到 RKNN 转换**：支持视觉和文本模型的批量转换
- **多平台支持**：针对 rk3566 等 Rockchip 平台进行优化
- **模型类型支持**：
  - CLIP 模型（视觉+文本双模型）
  - 人脸识别模型（检测+识别双模型）

### 2. CumSum 操作替换
- **自动检测**：识别 ONNX 模型中的 CumSum 操作节点
- **智能替换**：将不支持的 CumSum 操作替换为等效的 MatMul 操作
- **验证机制**：确保替换后的模型输出与原始模型一致

## 项目结构

```
immich_to_rknn2/
├── build_rknn.py              # 主要的 RKNN 转换脚本
├── convert.sh                 # 转换过程的封装脚本
├── process_all_models.py      # 批量处理多个模型
├── replace_cumsum.py          # CumSum 操作替换核心逻辑
├── check_cumsum.py           # 检测模型中的 CumSum 节点
├── verify_replacement.py     # 验证替换后的模型正确性
├── inspect_cumsum_detailed.py # 详细分析 CumSum 操作
├── requirements.txt          # Python 依赖
├── pyproject.toml           # 项目配置
├── nllb-clip-base-siglip__v1/    # NLLB CLIP 模型
│   ├── textual/model.onnx
│   └── visual/model.onnx
├── XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k/  # XLM-Roberta 模型
│   ├── textual/model.onnx
│   └── visual/model.onnx
├── test_cumsum/             # CumSum 替换后的测试输出
└── snapshot/                # 转换过程的分析快照
```

## 环境要求

### 系统依赖
```bash
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
```

### Python 环境
- Python 3.12+
- 主要依赖：
  - `rknn-toolkit2 >= 2.3.0` - RKNN 转换工具包
  - `onnx >= 1.16.1` - ONNX 模型处理
  - `numpy >= 1.26.4` - 数值计算
  - `setuptools >= 78.1.0` - 构建工具

## 使用方法

### 快速转换
```bash
# 基本转换命令
python3 build_rknn.py <模型目录> <目标平台>

# 示例：转换 NLLB CLIP 模型到 rk3566 平台
python3 build_rknn.py nllb-clip-base-siglip__v1/ rk3566

# 使用封装脚本（推荐）
./convert.sh nllb-clip-base-siglip__v1/ rk3566
```

### CumSum 操作处理
```bash
# 检查模型中的 CumSum 节点
python3 check_cumsum.py

# 替换 CumSum 操作
python3 replace_cumsum.py --input <输入模型> --output <输出模型>

# 批量处理所有模型
python3 process_all_models.py

# 验证替换结果
python3 verify_replacement.py <原始模型> <替换后模型>
```

## 技术细节

### RKNN 转换流程
1. **模型加载**：使用 RKNN API 加载 ONNX 模型
2. **平台配置**：设置目标 Rockchip 平台和优化参数
3. **模型构建**：执行量化和优化（可选择关闭量化）
4. **精度分析**：对文本模型进行精度分析
5. **导出 RKNN**：生成最终的 RKNN 模型文件

### CumSum 替换策略
- **检测机制**：遍历 ONNX 图节点，识别 CumSum 操作类型
- **替换逻辑**：使用 MatMul 操作实现等效的累积求和功能
- **参数处理**：正确处理 axis、reverse、exclusive 等属性
- **形状推断**：确保替换后的操作保持正确的张量形状

### 支持的模型架构
- **CLIP 模型**：支持视觉编码器和文本编码器
- **人脸模型**：支持检测分支和识别分支
- **动态输入**：支持配置动态输入尺寸

## 错误处理

### 常见问题
- **CumSum 不支持**：自动检测并提供替换方案
- **操作不支持**：日志中标记不支持的 ONNX 操作
- **精度问题**：提供精度分析工具和验证机制

### 日志分析
- **转换日志**：`immich_to_rknn2.log` 记录完整转换过程
- **错误分析**：`snapshot/error_analysis.txt` 详细分析错误
- **映射信息**：`snapshot/map_name_to_file.txt` 操作映射关系

## 开发说明

### 代码结构
- **模块化设计**：转换、检查、替换、验证功能分离
- **错误处理**：完善的异常处理和日志记录
- **批量处理**：支持多个模型的自动化处理

### 扩展性
- **自定义操作**：支持注册自定义 ONNX 操作
- **平台适配**：易于添加新的 Rockchip 平台支持
- **模型类型**：可扩展支持更多模型架构

## 注意事项

1. **内存管理**：RKNN 转换过程需要较多内存，建议在大内存环境中运行
2. **模型兼容性**：并非所有 ONNX 操作都被 RKNN 支持，需要预处理
3. **精度验证**：转换后建议进行精度测试，确保模型性能
4. **平台特定**：生成的 RKNN 模型只能在特定 Rockchip 平台上运行

## 故障排除

### 转换失败
- 检查 ONNX 模型是否完整且格式正确
- 查看 `immich_to_rknn2.log` 中的详细错误信息
- 确认目标平台参数是否正确

### CumSum 替换问题
- 使用 `check_cumsum.py` 确认模型中是否存在 CumSum 操作
- 检查替换后的模型结构是否正确
- 使用 `verify_replacement.py` 验证模型输出的准确性

### 运行时错误
- 确认 RKNN 运行时环境已正确安装
- 检查模型输入数据的格式和尺寸
- 验证 NPU 驱动和固件版本兼容性