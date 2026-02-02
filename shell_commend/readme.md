# 用于保存一些cuda代码编写过程中会使用到的shell指令

kernprof_sh.sh 用于python代码的耗时逐行分析。

nsys_sh.sh 用于使用nvidia自带的nsys工具对模型的运行时间进行分析，分析得到的.nsys-rep文件能使用nsigjht-compute和nsight-systems软件打开。
这个两个工具是安装cuda时自带的，一般在/usr/loacl/cuda-**.*/bin/下，为ncu-ui和nsys-ui,注意nsys的版本，其生成的文件只能向上兼容。