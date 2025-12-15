import argparse
import os
import tempfile
import onnx
import numpy as np

from numpy import cumsum, max, exp, sum
from rknn.api.custom_op import get_node_attr

parser = argparse.ArgumentParser("RKNN model converting")
parser.add_argument("model", help="Path to ONNX model file, ex: models/ViT-B-32__openai/textual/model.onnx", type=str)
parser.add_argument("target_platform", help="target platform ex:rk3566", type=str)
parser.add_argument("--dynamic-input", help="Dynamic input specification for detection models", type=str, default=None)
args = parser.parse_args()


def generate_random_input_for_model(model_path):
    """
    根据 ONNX 模型的输入规范动态生成随机输入数据。
    返回一个临时 npy 文件路径的列表。
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_basename = os.path.basename(model_path)
    original_dir = os.getcwd()
    
    try:
        os.chdir(model_dir)
        model = onnx.load(model_basename)
        try:
            onnx.load_external_data_for_model(model, ".")
        except:
            pass
    finally:
        os.chdir(original_dir)
    
    # ONNX 数据类型映射到 numpy 数据类型
    onnx_dtype_to_numpy = {
        1: np.float32,    # FLOAT
        2: np.uint8,      # UINT8
        3: np.int8,       # INT8
        4: np.uint16,     # UINT16
        5: np.int16,      # INT16
        6: np.int32,      # INT32
        7: np.int64,      # INT64
        9: np.bool_,      # BOOL
        10: np.float16,   # FLOAT16
        11: np.float64,   # DOUBLE
        12: np.uint32,    # UINT32
        13: np.uint64,    # UINT64
    }
    
    input_files = []
    temp_dir = tempfile.mkdtemp()
    
    for i, inp in enumerate(model.graph.input):
        # 获取输入的形状
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                # 动态维度，使用默认值
                shape.append(1)
            else:
                shape.append(1)
        
        # 获取数据类型
        elem_type = inp.type.tensor_type.elem_type
        numpy_dtype = onnx_dtype_to_numpy.get(elem_type, np.float32)
        
        # 生成随机数据
        if numpy_dtype in [np.int32, np.int64, np.int8, np.int16]:
            random_data = np.random.randint(0, 100, size=shape, dtype=numpy_dtype)
        elif numpy_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            random_data = np.random.randint(0, 100, size=shape, dtype=numpy_dtype)
        elif numpy_dtype == np.bool_:
            random_data = np.random.choice([True, False], size=shape)
        else:
            random_data = np.random.randn(*shape).astype(numpy_dtype)
        
        # 保存到临时文件
        temp_file = os.path.join(temp_dir, f"input_{i}_{inp.name.replace('/', '_')}.npy")
        np.save(temp_file, random_data)
        input_files.append(temp_file)
        print(f"Generated random input for '{inp.name}': shape={shape}, dtype={numpy_dtype}")
    
    return input_files


def ConvertModel(model_path='ViT-B-32__openai/textual/model.onnx', target_platform='rk3566', dynamic_input = None):
    # E build: Repeat call the 'rknn.build' or 'rknn.hybrid_quantization_step1' is not allow!
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)

    rknn.config(target_platform=target_platform, dynamic_input=dynamic_input, disable_rules=['fuse_matmul_softmax_matmul_to_sdpa'])

    onnx_to_load = model_path
    # if 1:
    #     ret = rknn.reg_custom_op(CumSum())

    #     if ret != 0:
    #         raise RuntimeError("Register Custom OP failed!")

    print(f"RKNN is loading ONNX :{onnx_to_load}")
    ret = rknn.load_onnx(model=onnx_to_load)

    if ret != 0:
        print("Load failed!")
        exit(ret)

    ret = rknn.build(do_quantization=False)

    print(f"RKNN build reported with status {ret}")
    if ret != 0:
        print("Build failed!")
        exit(ret)
    print(model_path.replace('model.onnx',f'{target_platform}.rknn'))
    if "textual" in model_path:
        input_files = generate_random_input_for_model(model_path)
        ret = rknn.accuracy_analysis(inputs=input_files)
        # 清理临时文件
        for f in input_files:
            if os.path.exists(f):
                os.remove(f)
        if ret != 0:
            print("Accuracy analysis failed!")
            exit(ret)
    ret = rknn.export_rknn(model_path.replace('model.onnx',f'{target_platform}.rknn'))
    print(f"RKNN export reported with status {ret}")
    if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
    print('done')
    del rknn
    del RKNN

    if not os.path.isfile(f'{model_path.replace("onnx","rknn")}'):
        print(f'Dummy model not found at {model_path.replace("onnx","rknn")}, creating one')
        with open(f'{model_path.replace("onnx","rknn")}', 'w'):
            pass


if not os.path.isfile(args.model):
    print(f'Error: Model file {args.model} not found!')
    exit(1)

dynamic_input = None
if args.dynamic_input:
    import json
    dynamic_input = json.loads(args.dynamic_input)

ConvertModel(model_path=args.model, target_platform=args.target_platform, dynamic_input=dynamic_input)
