# 采集推理程序的系统trace，生成report.qdrep
nsys profile -t cuda,nvtx --force-overwrite true -o ./gpu_cuda_test/profile/model_infer_int8_wbias.nsys-rep python ./gpu_cuda_test/model_infer_int8_wbias.py
# 用Nsight Systems打开report.qdrep，查看GPU kernel、数据传输、同步的时间线
