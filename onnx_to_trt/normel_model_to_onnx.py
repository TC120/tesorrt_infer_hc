"""
用于将已经得到的onnx模型转换为量化后的trt模型,注意转换为变尺寸或在变batch模型时，需要onnx转换时就支持。
"""

import os
import calibrator_int8
import numpy as np
import tensorrt as trt
import torch as t
from cuda import cudart

def one_input_onnx_to_trt(input_shape=(1, 3, 1920, 1080), 
                          min_input_shape=None,
                          max_input_shape=None,
                          type_model="float32",
                          onnx_path="./onnx_model/mobilenet_v3_small.onnx",
                          trt_path="./trt_model/mobilenet_v3_small.trt",
                          nCalibration = 5,
                          calibrationDataPath = "E:/data/image/int8_caibration/720p/",
                          int8_cache_file="./int8_cache_file/mobilenet_v3_small.cache"

                          ):
    #   固定随机源
    np.random.seed(31193)
    t.manual_seed(97)
    t.cuda.manual_seed_all(97)
    # 由于 CuDNN 为了提高计算效率，在某些情况下会采用非确定性的算法，这可能导致即使输入和模型参数都相同，每次运行的结果也会略有不同。True 时，CuDNN 会使用确定性算法。
    t.backends.cudnn.deterministic = True
    # print(1)
    #   设置参数
    if type_model == "float32":
        bUseFP16Mode = False
        bUseINT8Mode = False
    elif type_model == "float16":
        # for FP16 mode
        bUseFP16Mode = True
        bUseINT8Mode = False
    elif type_model == "int8":
        # for INT8 model
        bUseFP16Mode = False
        bUseINT8Mode = True
    else:
        print("type_model should be float32, float16, int8")
        exit()

    #   控制python中小数的显示精度
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()

    # Parse network, rebuild network and do inference in TensorRT ------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if bUseINT8Mode:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator_int8.MyCalibrator(calibrationDataPath, nCalibration, input_shape, int8_cache_file.format(type_model))

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)

    # 输入名：onnx生成时设置， 第一形状参数：最小形状，第二形状参数：最优形状，第三形状参数：最大形状。
    if min_input_shape is None or max_input_shape is None:
        
        profile.set_shape(inputTensor.name, input_shape, input_shape, input_shape)
    else:
        profile.set_shape(inputTensor.name, min_input_shape, input_shape, max_input_shape)

    
    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)

    if engineString is None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    with open(trt_path.format(type_model), "wb") as f:
        f.write(engineString)


if __name__=="__main__":
    one_input_onnx_to_trt(input_shape=(3, 3, 640, 640), 
                          type_model="int8", 
                          calibrationDataPath="/media/hc/DataDisk/data/Rally/detect/yolo11_cls_20260110/images/train/",
                          onnx_path="./onnx_model/yolo11n_cls_train_12_3x640_wnms_bias.onnx",
                          trt_path="./trt_model/yolo11n_cls_train_12_3x640_wnms_bias_{}.trt",
                          nCalibration = 700,
                          int8_cache_file="./int8_cache_file/yolo11n_cls_train_12_3x640_wnms_bias_{}.cache")

