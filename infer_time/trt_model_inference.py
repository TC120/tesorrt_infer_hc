import tensorrt as trt
import torch
import time
import numpy as np
import cv2
import pycuda.driver as cuda
from cuda import cudart
# import pycuda.autoinit
import common

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


# 加载tensorrt引擎文件
def load_engine(trt_path):
    # 反序列化引擎
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def trt_inference(engine_path, image_path):
    # input_image, h_, w_ = pre_processing(image_path)
    input_image = cv2.imread(image_path).astype(np.uint8)
    trt_engine = load_engine(engine_path)
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
    a = trt_inference("./trt_model/mobilenet_v3_smallint8.trt",
                      "/media/hc/DataDisk/data/Rally/detect/int8_calinrator_data_224/0000.png")
    print(a[0])

