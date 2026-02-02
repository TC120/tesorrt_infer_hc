import cv2
import torch
from torchvision.ops import nms
import json
import numpy as np
from cuda.bindings import driver as cuda, runtime as cudart, nvrtc
import tensorrt as trt
import matplotlib.pyplot as plt
import common
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def cuda_call(call):
    """Helper function to make CUDA calls and check for errors"""
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    err, res = call[0], call[1:]
    if err.value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                err.value, _cudaGetErrorEnum(err)
            )
        )
    if len(res) == 1:
        return res[0]
    elif len(res) == 0:
        return None
    else:
        return res
    


def wrap_gpu_ptr_to_tensor(gpu_ptr, device_id, size_in_bytes, dtype, shape):
    """
    将外部 GPU 指针的数据拷贝到 PyTorch Tensor 中
    
    Args:
        gpu_ptr: int，外部显存地址 (如 bindings[1])
        device_id: int，GPU 设备 ID
        size_in_bytes: int，数据总字节数 (如 t_size)
        dtype: torch.dtype，目标数据类型
        shape: tuple，目标形状
    """
    # 1. 在 PyTorch 中分配一个新的空 Tensor (显存已分配)
    # 注意：这里我们分配在 'cuda' 上
    tensor = torch.empty(size_in_bytes // 4, dtype=dtype, device=device_id)
    
    # 2. 执行 Device-to-Device 的内存拷贝
    # 将外部数据 (gpu_ptr) 拷贝到 PyTorch Tensor (tensor.data_ptr()) 中
    # 注意：确保 size_in_bytes 与 tensor 的总字节数一致
    error = cudart.cudaMemcpy(
        tensor.data_ptr(),          # 目标地址
        gpu_ptr,                    # 源地址
        size_in_bytes,              # 字节大小
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice # 关键：从显存到显存
    )
        
    return tensor.reshape(shape)
    

import random
def draw_detections_batch(
    images,                  # (N, H, W, 3) numpy数组或list
    detections_list,         # 检测结果列表，每个元素是(N_i, 6)的tensor
    class_names=None,        # 类别名称列表，如 ['person', 'car', 'dog']
    conf_threshold=0.25,    # 置信度阈值
    line_thickness=2,
    font_scale=0.5,
    return_numpy=False      # 是否返回numpy数组
):
    """
    将检测结果绘制到批量图片上
    
    Args:
        images: 图片数据，可以是:
               - numpy数组 (N, H, W, 3)
               - list of numpy arrays
        detections_list: 检测结果列表，每个元素是 (N_i, 6) 的tensor
        class_names: 类别名称列表
        conf_threshold: 置信度阈值
        line_thickness: 线条粗细
        font_scale: 字体大小
        return_numpy: True返回numpy数组，False返回PIL图片
    
    Returns:
        绘制后的图片列表
    """
    # 统一处理图片输入格式
    if isinstance(images, np.ndarray) and len(images.shape) == 4:
        # 如果是(N, H, W, 3)的numpy数组
        images = [img.copy() for img in images]
    elif isinstance(images, list):
        images = [img.copy() for img in images]
    elif isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
        if len(images.shape) == 4:
            images = [img.transpose(1, 2, 0) for img in images]
        else:
            images = [images]
    else:
        raise ValueError("不支持的图片格式")
    
    # 生成颜色（为每个类别生成不同的颜色）
    num_classes = len(class_names) if class_names else 80
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]
    
    annotated_images = []
    
    # 遍历每个batch
    for img_idx, (img, detections) in enumerate(zip(images, detections_list)):
        # 转换为BGR格式（OpenCV默认）
        # if len(img.shape) == 3 and img.shape[2] == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img_h, img_w = img.shape[:2]
        
        # 遍历每个检测框
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # 过滤低置信度
            if conf < conf_threshold:
                continue
            
            cls = int(cls)
            x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf)
            # x, y, w, h, conf, cls = det
            
            # # 过滤低置信度
            # if conf < conf_threshold:
            #     continue
            
            # cls = int(cls)
            # x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)
            
            # # 转换YOLO格式（中心点x,y + 宽高）到边界框格式（x1,y1,x2,y2）
            # x1 = int((x - w / 2) * 1)
            # y1 = int((y - h / 2) * 1)
            # x2 = int((x + w / 2) * 1)
            # y2 = int((y + h / 2) * 1)
            
            # 获取颜色
            color = colors[cls % len(colors)]
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # 准备标签文本
            label = f"{class_names[cls] if class_names else cls} {conf:.2f}" if class_names else f"{conf:.2f}"
            
            # 计算标签背景框的大小
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )
            
            # 绘制标签文字
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        
        # 转换回RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_images.append(img)
    
    return annotated_images if return_numpy else annotated_images

    


def separate_batch_results(nms_output, score_threshold=0.0):
    """
    将 NMS 输出拆分为每张图独立的检测结果列表，并去除无效填充框。
    
    Args:
        nms_output: (3, 300, 6) 的 Tensor
        score_threshold: 分数阈值，用于过滤掉填充的0框（默认0表示保留所有非0框）
    
    Returns:
        list: 包含3个 Tensor 的列表，每个 Tensor 形状为 (N_i, 6)
              例如: [Tensor(5个框, 6), Tensor(2个框, 6), Tensor(10个框, 6)]
    """
    batch_size = nms_output.shape[0]
    batch_results = []

    for i in range(batch_size):
        # 1. 提取当前 batch 的数据 (300, 6)
        single_batch = nms_output[i]
        
        # 2. 过滤：保留分数大于阈值的框
        # NMS 输出中未使用的位置通常是分数为0的，所以这里可以过滤掉
        mask = single_batch[:, 4] > score_threshold
        valid_dets = single_batch[mask]
        
        # 3. (可选) 按分数从高到低排序
        if len(valid_dets) > 0:
            valid_dets = valid_dets[valid_dets[:, 4].sort(descending=True).indices]
        
        # 4. 加入结果列表
        batch_results.append(valid_dets)

    return batch_results

    

def nms_postprocess(
    pred: torch.Tensor,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    class_agnostic: bool = False,
    num_classes: int = 3  # 根据你的7-4=3来设置
) -> torch.Tensor:
    """
    对推理结果进行NMS后处理
    
    Args:
        pred: 原始预测张量，形状 (1, 7, 6400) -> (batch, 4+num_classes, num_anchors)
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        max_det: 每张图最大检测数
        class_agnostic: 是否进行类别无关的NMS（跨类别抑制）
        num_classes: 类别数量 (7-4=3)
    
    Returns:
        NMS后的检测结果，形状 (1, max_det, 6) -> (box_coords(4) + score(1) + class(1))
    """
    # 转置: (1, 7, 6400) -> (1, 6400, 7)
    pred = pred.transpose(-1, -2)
    bs, num_anchors, channels = pred.shape
    
    # 分离坐标、分数和额外信息
    boxes = pred[..., :4]           # (1, 6400, 4)
    scores = pred[..., 4:4+num_classes]  # (1, 6400, 3)
    extra_shape = channels - (4 + num_classes)
    extras = pred[..., 4+num_classes:] if extra_shape > 0 else None  # 额外信息（如mask系数等）
    
    # 找最大分数和对应类别
    max_scores, classes = scores.max(dim=-1)  # (1, 6400)
    
    # 准备输出张量
    output = torch.zeros(bs, max_det, 6, device=pred.device, dtype=pred.dtype)
    
    for i in range(bs):
        # 提取当前batch的预测
        box = boxes[i]        # (6400, 4)
        score = max_scores[i]  # (6400,)
        cls = classes[i]      # (6400,)
        extra = extras[i] if extras is not None else None
        
        # 置信度过滤
        mask = score > conf_threshold
        if not mask.any():
            continue  # 没有满足阈值的框
            
        box = box[mask]
        score = score[mask]
        cls = cls[mask]
        if extra is not None:
            extra = extra[mask]
        
        # NMS处理
        if class_agnostic:
            # 类别无关：所有类别一起做NMS
            keep = nms(box, score, iou_threshold)[:max_det]
        else:
            # 类别相关：每个类别单独做NMS
            keep = []
            for c in range(num_classes):
                class_mask = cls == c
                if not class_mask.any():
                    continue
                
                class_boxes = box[class_mask]
                class_scores = score[class_mask]
                class_keep = nms(class_boxes, class_scores, iou_threshold)
                
                # 添加类别索引到keep
                original_indices = torch.where(class_mask)[0][class_keep]
                keep.append(original_indices)
            
            # 合并所有类别的结果并按分数排序
            if keep:
                keep = torch.cat(keep)
                keep = keep[score[keep].sort(descending=True).indices][:max_det]
            else:
                keep = torch.tensor([], dtype=torch.long, device=pred.device)
        
        # 构建最终检测结果
        dets = torch.cat([
            box[keep],
            score[keep].unsqueeze(1),
            cls[keep].unsqueeze(1).to(pred.dtype)
        ], dim=-1)  # (N, 6)
        
        # 填充到输出张量
        num_dets = min(dets.shape[0], max_det)
        output[i, :num_dets] = dets[:num_dets]
    
    return output


    

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

    

def laod_data_test(path, nums, images_size=[640, 640, 3]):
    h, w, c = images_size
    img_array = np.zeros((nums, h, w, c))
    for i in range(nums):
        img_array[i] = cv2.imread(f"{path}/{i:04d}.png")

    # return img_array
    return img_array.astype(np.uint8)


@profile
def gpu_inference_0(img_np, bindings, context, device, stream, t_size, trt_infer_type):
    ## 前处理部分
    tensor = torch.from_numpy(img_np).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = tensor / 255.0
    gpu_tensor = tensor.contiguous()
    tensor_ptr = gpu_tensor.data_ptr()
    tensor_size_bytes = tensor.nbytes
    ## 模型推理部分
    cudart.cudaMemcpy(bindings[0], tensor_ptr, tensor_size_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    context.execute_async_v3(stream_handle=stream)
    after_infer_data = wrap_gpu_ptr_to_tensor(bindings[1], device, t_size, torch.float32, (3, 300, 6))
    after_infer_data = separate_batch_results(after_infer_data)

    # 将数据从gpu上读取到cpu并处理
    # host_buffer = torch.zeros(t_size // 4, dtype=torch.float32, pin_memory=True)
    # cuda_call(cudart.cudaMemcpy(host_buffer.data_ptr(), bindings[1], t_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    # output = host_buffer.reshape(3, 300, 6)
    # output = separate_batch_results(output)
    # return output
    return after_infer_data


@profile
def gpu_inference(img_np, bindings, context, device, stream, t_size, trt_infer_type):
    tensor = torch.from_numpy(img_np).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = tensor / 255.0
    gpu_tensor = tensor.contiguous()
    tensor_ptr = gpu_tensor.data_ptr()
    tensor_size_bytes = tensor.nbytes


    cudart.cudaMemcpy(bindings[0], tensor_ptr, tensor_size_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    context.execute_async_v3(stream_handle=stream)
    after_infer_data = wrap_gpu_ptr_to_tensor(bindings[1], device, t_size, torch.float32, (3, 300, 6))
    after_infer_data = separate_batch_results(after_infer_data)
    # 将数据直接在gpu上处理
    # after_infer_data = wrap_gpu_ptr_to_tensor(bindings[1], device, t_size, torch.float32, (3, 7, 8400))
    # if trt_infer_type == "int8":
    #     after_infer_data[:, 4:, :].add_(3)
    # after_infer_data[:, 4:, :] = torch.sigmoid(after_infer_data[:, 4:, :])
    
    # # 将数据从gpu上读取到cpu
    # host_buffer = torch.zeros(t_size // 4, dtype=torch.float32, pin_memory=True)
    # cuda_call(cudart.cudaMemcpy(host_buffer.data_ptr(), bindings[1], t_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    # output = host_buffer.reshape(3, 7, 8400)
    # if trt_infer_type == "int8":
    #     output[:, 4:, :].add_(3)
    # # 执行 Sigmoid
    # output[:, 4:, :] = torch.sigmoid(output[:, 4:, :])
    # # 执行nms
    # output = nms_postprocess(output)
    # output = separate_batch_results(output)
    # return after_infer_data
    return 0


@profile
def mecpy_gpu_tensor(trt_infer_type):
    # 测试 GPU Tensor 之间进行数据拷贝的函数使用
    batchs = 3
    test_frames = 11
    image_cpu = laod_data_test("/media/hc/DataDisk/data/Rally/detect/int8_calinrator_data_640", test_frames * batchs)
    trt_engine = load_engine_ultrulytics("/home/hc/pycharm_ws/tensorrt_infer_hc/trt_model/yolo11n_cls_train_12_3x640_wnms_float16.trt")
    trt_context = trt_engine.create_execution_context()
    tensor_names = [trt_engine.get_tensor_name(i) for i in range(trt_engine.num_io_tensors)]
    bindings = []
    tmp_size = None
    for binding in tensor_names:
        shape = trt_engine.get_tensor_shape(binding) 
        size = trt.volume(shape)
        trt_type = trt_engine.get_tensor_dtype(binding)
        dtype = np.dtype(trt.nptype(trt_type))
        tmp_size = size * dtype.itemsize
        tmp_ptr = cuda_call(cudart.cudaMalloc(size * dtype.itemsize))
        bindings.append(int(tmp_ptr))
        trt_context.set_tensor_address(binding, int(tmp_ptr))
    stream = cuda_call(cudart.cudaStreamCreate())
    device = torch.device("cuda:0")
    
    img_in_np = image_cpu[0:3]
    infer_result = gpu_inference_0(img_in_np, bindings, trt_context, device, stream, tmp_size, trt_infer_type)
    
    # for i in range(1, test_frames, 1):
    #     img_in_np = image_cpu[i*batchs:(i+1)*batchs]
    #     gpu_inference(img_in_np, bindings, trt_context, device, stream, tmp_size, trt_infer_type)

    class_names = ['person', 'ball', 'racket']  # 根据你的模型类别数修改

    # 绘制检测结果
    annotated_images = draw_detections_batch(
        images=img_in_np,           # 你的图片数据
        detections_list=infer_result,  # 检测结果列表
        class_names=class_names,      # 类别名称
        conf_threshold=0.5,          # 置信度阈值
        line_thickness=2
    )
    # 或者显示结果
    for i, img in enumerate(annotated_images):
        cv2.imshow(f'Image {i}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # mecpy_gpu_tensor("float16")
    mecpy_gpu_tensor("int8")






