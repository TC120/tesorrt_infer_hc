import onnx
import numpy as np
import onnx.numpy_helper as numpy_helper

def print_onnx_layers(onnx_model_path):
    """
    打印ONNX模型中所有层的名称和索引
    """
    # 加载模型
    model = onnx.load(onnx_model_path)
    
    print("=" * 80)
    print(f"模型: {onnx_model_path}")
    print(f"输入数量: {len(model.graph.input)}")
    print(f"输出数量: {len(model.graph.output)}")
    print("=" * 80)
    
    # 打印所有节点（层）
    print("\n📋 层信息 (节点):")
    print("-" * 80)
    for i, node in enumerate(model.graph.node):
        print(f"索引: {i:3d} | 名称: {node.name:30s} | 类型: {node.op_type:15s} | 输入: {node.input} | 输出: {node.output}")
    
    # 打印输入信息
    print("\n📥 模型输入:")
    print("-" * 80)
    for i, input in enumerate(model.graph.input):
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"输入 {i}: {input.name} | 形状: {shape}")
    
    # 打印输出信息  
    print("\n📤 模型输出:")
    print("-" * 80)
    for i, output in enumerate(model.graph.output):
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"输出 {i}: {output.name} | 形状: {shape}")


def print_onnx_constant(model_path, tensor_name):
    """
    直接从 ONNX 模型中读取并打印指定名称的常量/权重数据。
    """
    model = onnx.load(model_path)
    
    found = False
    # 1. 在 initializer 中查找 (最常见)
    for init in model.graph.initializer:
        if init.name == tensor_name:
            print(f"✅ 在 Initializer 中找到: {tensor_name}")
            tensor = numpy_helper.to_array(init)
            print(f"   类型: {tensor.dtype}")
            print(f"   形状: {tensor.shape}")
            print(f"   数值内容:\n{tensor}")
            found = True
            break
    
    # 2. 如果没在 initializer 找到，可能在 Constant 节点中 (较少见，但也可能)
    if not found:
        for node in model.graph.node:
            if node.op_type == "Constant" and node.output[0] == tensor_name:
                print(f"✅ 在 Constant 节点中找到: {tensor_name}")
                # 获取属性
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = numpy_helper.to_array(attr.t)
                        print(f"   类型: {tensor.dtype}")
                        print(f"   形状: {tensor.shape}")
                        print(f"   数值内容:\n{tensor}")
                        found = True
                        break
    
    if not found:
        print(f"❌ 未找到名为 '{tensor_name}' 的常量/权重。它可能是中间计算结果，请使用方法二。")


# 使用示例
print_onnx_layers("./onnx_model/yolo11n_cls_train_12_3x640_wnms_bias.onnx")
print_onnx_constant("./onnx_model/yolo11n_cls_train_12_3x640_wnms_bias.onnx", "/model/model.23/Constant_4_output_0")