import cv2
import torch
import numpy as np
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import common
import json
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


# 加载tensorrt引擎文件
def load_engine_ultrulytics(trt_path):
    '''
    反实例化ultrulytics生成的engine文件时，由于其官方会在开头写入一些他自己使用的数据，
    导致反实例化时tensorrt无法解析，可以按照
    https://blog.csdn.net/ogebgvictor/article/details/145858668
    的方案解决
    '''
    # 反序列化引擎
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
        except:
            f.seek(0)

        return runtime.deserialize_cuda_engine(f.read())


def preprocess_gpu(device, batch_images, resize_size):
    tensor = torch.from_numpy(batch_images).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=resize_size, mode='bicubic', align_corners=False)
    tensor = tensor / 255.0
    return tensor

# @profile
def img_infer():
    # 用于测试图片在cpu上进行预处理再进行数据迁移到GPU的时间
    test_frames = 10
    batchs = 3
    image_cpu = cv2.imread("/home/hc/pycharm_ws/tensorrt_infer_hc/image_data/img_160.png")
    # trt_engine = load_engine_ultrulytics("/home/hc/pycharm_ws/tensorrt_infer_hc/trt_model/yolo11n_cls_train_12_int8_batch3.engine")
    trt_engine = load_engine_ultrulytics("/home/hc/pycharm_ws/tensorrt_infer_hc/trt_model/yolo11n_cls_train_12_3x640_wnms_int8.trt")
    context = trt_engine.create_execution_context()
    # inputs 和 outputs中包括数据size大小，数据所占byte数，host和device的device指针地址
    inputs, outputs, bindings = common.allocate_buffers(trt_engine)
    # 将图片生成np数组
    img_np = np.zeros((batchs, 160, 160, 3), dtype=np.uint8)
    for i in range(batchs):
        img_np[i] = image_cpu

    resize_size = (640, 640)
    device = torch.device("cuda:0")
    gpu_tensor = preprocess_gpu(device, img_np, resize_size)
    for _ in range(test_frames):
        gpu_tensor = preprocess_gpu(device, img_np, resize_size)
    

    # 确保Tensor是连续的且在GPU上
    gpu_tensor = gpu_tensor.contiguous()
    assert gpu_tensor.is_cuda, "Tensor must be on GPU"
    with common.CudaStreamContext() as stream:
        tensor_ptr = gpu_tensor.data_ptr()
        cuda.memcpy_dtod(inputs[0].gpu_tensor.data_ptr(), inputs[0].size)
        # 设置TensorRT张量地址
        for i in range(trt_engine.num_io_tensors):
            name = trt_engine.get_tensor_name(i)
            context.set_tensor_address(name, bindings[i])
        
        # 执行推理
        context.execute_async_v3(stream_handle=stream.stream)
    
        
        trt_outputs = outputs[0].host.copy() 
        common.free_buffers(inputs, outputs)
    
    return trt_outputs


if __name__ == "__main__":
    a = img_infer()











