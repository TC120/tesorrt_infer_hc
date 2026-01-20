# 代码用于从pytorch获取官方模型. 然后将其转化为onnx
# 参考https://docs.pytorch.org/vision/main/models.html

from torchvision.models import (
    list_models,  # 列举可用模型
    get_model,    # 快速初始化模型
    get_weight,   # 获取权重对象
    get_model_weights  # 获取模型对应的权重枚举类
)
import torch.nn as nn
import torch


class BackboneWrapper(nn.Module):
    """骨干网络包装器，支持特征维度压缩到目标尺寸（默认1024）"""

    def __init__(
        self,
        backbone_in: nn.Module,
    ):
        """
        简单进行模型输出的下采样，然后展平为1维数据
        """
        super().__init__()
        self.backbone = backbone_in

        # 1. 初始化基础组件（池化+通道压缩+维度映射）
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 直接指定输出1x1


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 骨干网络提取特征
        x = self.backbone(x)  # (batch, in_channels, h, w)
        x = self.pool(x)  # (batch, in_channels, pooled_h, pooled_w)
        x = x.reshape(x.shape[0], -1)  # (batch, target_dim)

        return x


# 列举torchvision所有可用模型（约100+个，含分类、检测、分割等
def print_all_available_models() -> None:
    all_models = list_models() 
    for model in all_models:
        print(model)


# 获取指定名称的模型及其权重
def get_torch_vision_backbone(model_name: str) -> nn.Module:
    model = get_model(model_name, weights="DEFAULT")
    model.eval()
    backbone = model.features
    return backbone


if __name__ == "__main__":
    # 查看所有可以使用的backbone
    print_all_available_models()

    mobilenet_v3_small_backbone = get_torch_vision_backbone("mobilenet_v3_small")
    new_model = BackboneWrapper(
        backbone_in=mobilenet_v3_small_backbone,
    )
    new_model.eval()

    torch.save(new_model, './pt_model/mobilenet_v3_small.pth')

    # 4. 定义导出参数
    onnx_path = "./onnx_model/mobilenet_v3_small.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)  # 与模型输入尺寸一致（1张图，3通道，640x640）

    # 5. 导出 ONNX（支持动态 Batch 维度，方便批量推理）
    torch.onnx.export(
        model=new_model,
        args=dummy_input,
        f=onnx_path,
        input_names=["input"],  # 输入名称（后续推理用）
        output_names=["output"],  # 输出名称（后续推理用）
        dynamic_axes={
            "input": {0: "batch_size"},  # 动态 Batch 维度（支持任意 batch 大小）
            "output": {0: "batch_size"}
        },
        opset_version=12,  # 兼容大部分 ONNX Runtime 版本
        do_constant_folding=True  # 优化常量折叠，提升推理速度
    )

