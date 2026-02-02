import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import onnx.numpy_helper as numpy_helper
import numpy as np

def remove_sigmoid_and_remap_output(onnx_path, output_path):
    # 加载原始模型
    model = onnx.load(onnx_path)
    
    # 获取图信息
    graph = model.graph
    nodes = graph.node

    
    # 查找目标节点, 这里的修改涉及3个节点之间的数据户通
    split_node = None
    sigmoid_node = None
    concat_node = None
    
    for node in nodes:
        if node.name == "/model.23/Split":
            split_node = node
        elif node.name == "/model.23/Sigmoid":
            sigmoid_node = node
        elif node.name == "/model.23/Concat_5":
            concat_node = node
    
    if not all([sigmoid_node, split_node, concat_node]):
        raise ValueError("未能找到所有目标节点")
    
    # 获取Sigmoid的输入（即Concat的输入之一）
    sigmoid_input = sigmoid_node.input[0]
    sigmoid_output = sigmoid_node.output[0]

    # 获取concat的输入, 同时判断concat需要被改变的输入序号
    concat_intput = concat_node.input
    change_idx = 0
    i = 0
    for node in concat_intput:
        if node == sigmoid_output:
            change_idx = i
        else:
            i+=1
    # 直接将sigmod的输入(split的输出)链接到concat的输入
    concat_node.input[change_idx] = sigmoid_input

    # 从图中移除节点
    graph.node.remove(sigmoid_node)
    
    
    # 运行形状推断
    model = shape_inference.infer_shapes(model)
    
    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"模型已保存到: {output_path}")



def add_fixed_bias_to_onnx(input_onnx_path, output_onnx_path):
    # 1. 加载模型
    model = onnx.load(input_onnx_path)
    
    # 2. 查找目标节点: /model/model.23/Sigmoid
    sigmoid_node = None
    for node in model.graph.node:
        if node.name == "/model/model.23/Sigmoid":
            sigmoid_node = node
            break
    
    if sigmoid_node is None:
        print("错误：找不到节点 /model/model.23/Sigmoid")
        return

    # 获取 Sigmoid 的原始输入名称
    original_input_name = sigmoid_node.input[0]
    
    # 定义新节点的名称和输出名称
    bias_name = "fixed_bias_b"  # 这是常量的名称，存在于模型内部
    add_node_name = "Add_Fixed_Bias_Node"
    add_output_name = "biased_tensor_output"

    # 3. 创建常量张量
    # 使用 helper.make_tensor 创建一个包含固定值的张量
    # data_type 这里设置为 FLOAT (float32)，如果你的模型是 double/float16，请相应修改
    bias_tensor = helper.make_tensor(
        name=bias_name,
        data_type=TensorProto.FLOAT, 
        dims=[], # 标量，形状为空
        vals=[2.5] # 固定值 b=2
    )
    
    # 将常量添加到图的 initializers 中，而不是 inputs 中
    model.graph.initializer.append(bias_tensor)

    # 4. 创建 Add 节点
    # Add 节点的输入：一个是上游数据流，一个是刚才定义的常量名称
    add_node = helper.make_node(
        'Add',
        inputs=[original_input_name, bias_name],
        outputs=[add_output_name],
        name=add_node_name
    )

    # 5. 修改 Sigmoid 节点的输入，将其指向 Add 节点的输出
    sigmoid_node.input[0] = add_output_name

    # 6. 将 Add 节点插入到图中 (插在 Sigmoid 节点之前)
    node_idx = 0
    for i, node in enumerate(model.graph.node):
        if node.name == sigmoid_node.name:
            node_idx = i
            break
    
    model.graph.node.insert(node_idx, add_node)

    # 7. 检查模型并保存
    try:
        onnx.checker.check_model(model)
        print("模型检查通过，保存中...")
        onnx.save(model, output_onnx_path)
        print(f"修改后的模型（已包含固定偏置）已保存至: {output_onnx_path}")
    except Exception as e:
        print(f"模型检查失败: {e}")

if __name__=="__main__":
    # 使用示例
    # remove_sigmoid_and_remap_output("./onnx_model/yolo11n_cls_train_12_3x640_wonms.onnx", 
    #                                 "./onnx_model/yolo11n_cls_train_12_3x640_wonms_wosigmod.onnx")
    # add_fixed_bias_to_onnx("./onnx_model/yolo11n_cls_train_12_3x640_wnms.onnx", 
    #                        "./onnx_model/yolo11n_cls_train_12_3x640_wnms_bias.onnx")
    
