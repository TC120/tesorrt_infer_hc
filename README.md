# tesorrt_infer_hc
用于暂时存放tensorrt相关的pt模型转onnx再转trt的量化推理相关python代码。适用于cuda12.9, tensorrt10.4.1版本
代码来源于tensorrt官方代码与个人提供者的拼接整合用于自己使用。官方代码更加合规可信但是例子较少。
https://github.com/NVIDIA/TensorRT.git




前期环境配置要注意，如果只安装了python版本的tensorrt，是没有trtexec工具的，所有的转换需要依靠python代码来实现。注意cuda-python(from cuda import cudart)的安装需要指定为自己的cuda版本，严格匹配。否则无法使用, 
pip install cuda-python==12.9。
如果还通过deb的方式安装了C++版本的trt，是可以使用trtexec的，但是注意其与python版本是否一致，否则两者即使在同一硬件中，生成的文件也无法互相使用。


##  onnx_model_generator
    在torch_model_to_onnx.py中，使用torch进行模型选择和下载，得到.pt格式的模型文件，并且对模型效果进行简单验证。
    

##  onnx_to_trt
    onnx_to_trt.py中，提供了两种转化方案，one_input_onnx_to_trt()作为单输入的例子。mult_input_onnx_to_trt()（未实现）
    可以使用calibrator_int8.py来在int8量化时提供量化数据。


##  infer_time
    在trt_model_inferrence.py中，使用tensorrt和cuda-python，pycuda等库来加载trt模型并推理。
    trt_inference()中仅仅实现了推理的过程，没有使用前后处理等过程。



