common.py和common_runtime.py文件是cuda官方代码，用于创建cuda流和cuda中使用的内存空间。

video_cuda_time.py测试了cv2的cuda算子和cpu算子的运行速度区别。

preprocess_compare.py 对比了模型前处理在cpu和gpu上的处理的时间区别。

model_infer_*.py系列用于推理int8和fp16的yolo11n模型，其包括前处理和后处理。
*_int8.py将模型的后处理用自己代码实现代码，将nms放在模型外。
*_int8_wbias.py将模型的的nms放到模型中。
