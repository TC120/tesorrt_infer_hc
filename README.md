# tensorrt_infer_hc
本项目用于存放基于 TensorRT 实现的模型量化推理相关 Python 代码，核心流程为「PyTorch 模型 → ONNX 模型 → TensorRT 引擎」，适配 **CUDA 12.9 + TensorRT 10.4.1** 版本，仅用于个人学习与使用。

> 代码说明：核心逻辑基于 NVIDIA TensorRT 官方示例代码改造，结合个人业务需求整合；官方代码合规性更强但示例场景有限，本项目补充了量化、多输入（待实现）等实用场景。
> 官方仓库：https://github.com/NVIDIA/TensorRT.git

## 环境配置（关键！）
### 1. 核心依赖安装
| 依赖库         | 安装命令（适配 CUDA 12.9）| 说明                     |
|----------------|---------------------------------|--------------------------|
| TensorRT (Python) | `pip install tensorrt==10.4.1`  | 核心推理/转换库          |
| cuda-python    | `pip install cuda-python==12.9` | 严格匹配 CUDA 版本，否则导入 `cudart` 失败 |
| pycuda         | `pip install pycuda`            | CUDA 内存/流管理依赖     |
| PyTorch        | 适配 CUDA 12.9 的稳定版本       | 模型导出 ONNX 需用       |
| OpenCV-Python  | `pip install opencv-python`     | 图片预处理/后处理        |

### 2. 关键注意事项
- **trtexec 工具可用性**：
  - 仅安装 Python 版 TensorRT 时，无 `trtexec` 工具，需完全通过 Python 代码实现 ONNX→TRT 转换；
  - 若通过 deb 包安装 C++ 版 TensorRT，可使用 `trtexec`，但需保证 C++ 版本与 Python 版本完全一致（否则生成的 TRT 引擎文件无法跨版本使用）。
- CUDA 版本匹配：`cuda-python` 版本必须与系统 CUDA 驱动版本（`nvcc -V` 查看）严格一致，否则会出现 `ImportError: cannot import name 'cudart'` 等错误。

## 目录结构与功能说明
### gpu_cuda_test
- 用于测试模型的前后处理以及模型推理的GPU上耗时估计。

### gpu_cuda_test
- 用于测试模型的前后处理以及模型推理的GPU上耗时估计。


### onnx_model_generator
- 核心脚本：`torch_model_to_onnx.py`
- 功能：
  - 将 .pt 模型导出为 ONNX 格式（TensorRT 兼容）。
- 核心脚本：`onnx_model_change.py`
- 功能：
  - 对onnx模型的层进行删除与添加

### onnx_to_trt
- 核心脚本：`onnx_to_trt.py`
  - `one_input_onnx_to_trt()`：单输入 ONNX 模型转 TensorRT 引擎的完整示例（支持 FP32/FP16/INT8 量化）；
  - `mult_input_onnx_to_trt()`：多输入模型转换函数（待实现）；
- 辅助脚本：`calibrator_int8.py`
  - 为 INT8 量化提供校准数据集，解决低精度量化精度损失问题。

### infer_time
- 核心脚本：`trt_model_inference.py`
  - 功能：加载已生成的 TensorRT 引擎文件，实现模型推理（仅核心推理流程，暂未集成完整前后处理）；
  - 关键函数：`trt_inference(engine_path, image_path)` —— 输入引擎路径+图片路径，返回推理结果。

### shell_commend
- 用于ubuntu中nvida的trt软件与nsys分析软件的shell调用方法。