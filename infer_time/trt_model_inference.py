import tensorrt as trt
import torch
import time
import numpy as np
import cv2
import pycuda.driver as cuda
from cuda import cudart
# import pycuda.autoinit
import common
import json

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

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


# 加载tensorrt引擎文件
def load_engine(trt_path):
    # 反序列化引擎
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def trt_inference(engine_path, image_path):
    # input_image, h_, w_ = pre_processing(image_path)
    input_image = cv2.imread(image_path).astype(np.uint8)
    trt_engine = load_engine_ultrulytics(engine_path)
    context = trt_engine.create_execution_context()

    inputs, outputs, bindings = common.allocate_buffers(trt_engine)

    with common.CudaStreamContext() as stream:
        # Do inference
            print("Running inference on image {}...".format(image_path))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = input_image
            trt_outputs = common.do_inference(
                context,
                engine=trt_engine,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            
            # Free host and device memory used for inputs and outputs
            common.free_buffers(inputs, outputs)
    print(0)
    return trt_outputs


if __name__ == "__main__":
    a = trt_inference("/home/hc/pycharm_ws/tensorrt_infer_hc/trt_model/yolo11n_cls_train_12_3x640_wnms_float16.trt",
                      "/home/hc/pycharm_ws/tensorrt_infer_hc/image_data/img_640.png")
    print(a[:10])
    print(a[1763:])

