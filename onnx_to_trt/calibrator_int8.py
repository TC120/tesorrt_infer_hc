#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart


class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def  __init__(self, calibrationDataPath, nCalibration, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList = glob(calibrationDataPath + "*.png")[:10]
        self.nCalibration = nCalibration
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)

        for i in range(self.shape[0]):
            image_data = cv2.imread(imageList[i]).astype(np.float32)
            image_data = image_data.transpose(2, 1, 0)
            if np.max(image_data) > 256:  # 16-bit image
                max_range = 65535
                print('\tInput is a 16-bit image')
            else:
                max_range = 255
            data = image_data / max_range
            res[i] = data
        return res

    def get_batch_size(self):  # necessary API
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # necessary API
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return
