import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import sys


# @profile
def test_cpu_vs_gpu_resize():
    test_frames = 1000
    image_cpu = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    image_cpu = image_cpu[460:620, 880:1040, :]
    resize_size = (640, 640)
    image_gpu = cv2.cuda_GpuMat()
    image_gpu.upload(image_cpu)


    # ===== 测试CPU resize耗时 =====
    for _ in range(test_frames):
        frame_resized = cv2.resize(image_cpu, resize_size, interpolation=cv2.INTER_CUBIC)

    # ===== 测试GPU resize耗时 =====
    for _ in range(test_frames):
        gpu_resized = cv2.cuda.resize(image_gpu, resize_size, interpolation=cv2.INTER_CUBIC)



# 这里直接使用的是yolo官方使用前后处理方案
# @profile
def preprocess(device, im) -> torch.Tensor:
    """
    Prepare input image before inference.

    Args:
        im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

    Returns:
        (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.half() 
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


# @profile
def preprocess_0(device, im) -> torch.Tensor:
    """
    Prepare input image before inference.

    Args:
        im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

    Returns:
        (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.half() 
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


# @profile
def preprocess_gpu(device, batch_images, resize_size):
    # tensor = torch.from_numpy(batch_images).to(device, non_blocking=True)  # 保持uint8传输节省带宽
    # tensor = tensor.permute(0, 3, 1, 2).float()         # 在GPU上转换类型
    # tensor = F.interpolate(tensor, size=(640, 640), mode='bilinear', align_corners=False)
    # tensor = tensor / 255.0
    # return tensor

    tensor = torch.from_numpy(batch_images).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=resize_size, mode='bicubic', align_corners=False)
    tensor = tensor / 255.0
    return tensor

# @profile
def preprocess_gpu_0(device, batch_images, resize_size):
    # tensor = torch.from_numpy(batch_images).to(device, non_blocking=True)  # 保持uint8传输节省带宽
    # tensor = tensor.permute(0, 3, 1, 2).float()         # 在GPU上转换类型
    # tensor = F.interpolate(tensor, size=(640, 640), mode='bilinear', align_corners=False)
    # tensor = tensor / 255.0
    # return tensor

    tensor = torch.from_numpy(batch_images).to(device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=resize_size, mode='bicubic', align_corners=False)
    tensor = tensor / 255.0
    return tensor



# @profile
def img_prerocess_test():
    test_frames = 10
    image_cpu = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    cut_image = image_cpu[460:620, 880:1040, :]
    resize_size = (640, 640)
    cut_image_resized_cpu = cv2.resize(cut_image, resize_size, interpolation=cv2.INTER_CUBIC)
    device = torch.device("cuda:0")
    # 将图片生成列表
    img_np = np.zeros((test_frames,640,640,3), dtype=np.uint8)
    for i in range(test_frames):
        img_np[i] = cut_image_resized_cpu
    preprocessed_imgs_list = preprocess_0(device, img_np)
    for _ in range(test_frames):
        preprocessed_imgs_list = preprocess(device, img_np)
    for _ in range(test_frames):
        preprocessed_imgs_list = preprocess_gpu(device, img_np)
    return 0


# @profile
def img_preprocess_cpu():
    # 用于测试图片在cpu上进行预处理再进行数据迁移到GPU的时间
    test_frames = 10
    batchs = 3
    image_cpu = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    cut_image = image_cpu[460:620, 880:1040, :]
    resize_size = (640, 640)
    resize_img_cpu = cv2.resize(cut_image, resize_size, interpolation=cv2.INTER_CUBIC)
    device = torch.device("cuda:0")
    # 将图片生成np数组
    img_np = np.zeros((batchs, 640, 640, 3), dtype=np.uint8)
    for i in range(batchs):
        img_np[i] = resize_img_cpu
    preprocessed_imgs_list = preprocess_0(device, img_np)
    for _ in range(test_frames):
        preprocessed_imgs_list = preprocess(device, img_np)
    return 0


# @profile
def img_preprocess_gpu():
    # 用于测试图片在cpu上进行预处理再进行数据迁移到GPU的时间
    test_frames = 10
    batchs = 3
    image_cpu = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    cut_image = image_cpu[460:620, 880:1040, :]
    # 将图片生成np数组
    img_np = np.zeros((batchs, 160, 160, 3), dtype=np.uint8)
    for i in range(batchs):
        img_np[i] = cut_image

    resize_size = (640, 640)
    device = torch.device("cuda:0")
    preprocessed_imgs_list = preprocess_gpu_0(device, img_np, resize_size)
    for _ in range(test_frames):
        preprocessed_imgs_list = preprocess_gpu(device, img_np, resize_size)


    return 0
    

if __name__ == "__main__":
    img_preprocess_gpu()



