import cv2


# @profile
def read_capture_in_cpu():
    cap = cv2.VideoCapture(0)

    # 打开摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 640))
            cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 检测到q键（小写）
                print("检测到q键，停止写入并退出")
                break
        else:
            print("图片读取失败")
            break
    
    cap.release()
    cv2.destroyAllWindows()


def ckeck_opencv_cuda_support():

    # 方法1：检查 CUDA 设备数量（核心验证）
    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"支持的 CUDA 设备数量：{cuda_device_count}")

    # 方法2：查看 OpenCV 编译信息（详细验证）
    print("\nOpenCV 编译配置：")
    print(cv2.getBuildInformation())

@profile
def read_capture_in_gpu():
    cap = cv2.VideoCapture(0)
    cuda_stream = cv2.cuda_Stream()
    # 打开摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame, cuda_stream)
            gpu_gray = cv2.cuda.resize(gpu_frame, (640, 640), stream=cuda_stream)
            # 3. 同步CUDA流：等待GPU操作完成后，再下载到CPU
            # cuda_stream.waitForCompletion()
            out_frame = gpu_gray.download(cuda_stream)
            cv2.imshow("Camera out_frame", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 检测到q键（小写）
                print("检测到q键，停止写入并退出")
                break
        else:
            print("图片读取失败")
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # read_capture_in_cpu()
    read_capture_in_gpu()









