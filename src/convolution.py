from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

from functions import Functions


class Convolution:
    def __init__(self,
                 kernelSize: int,
                 padding: int) -> None:

        self.__kernelSize = kernelSize
        self.__padding = padding
        pass

    def operator(self, input: np.ndarray) -> float:
        raise NotImplementedError("Operator is not implemented!!")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """畳み込みを実行するためのメソッドである

        Args:
            img (np.ndarray): 適用させる画像

        Returns:
            np.ndarray: 出力画像
        """

        inputImg = img.copy().astype(np.float32)
        nChannels = inputImg.shape[-1]

        if self.__padding > 0:
            inputImg = np.pad(inputImg,
                              ((self.__padding, self.__padding),
                               (self.__padding, self.__padding),
                               (0, 0)))

        output = np.empty(shape=(nChannels,
                                 inputImg.shape[0]-self.__kernelSize+1,
                                 inputImg.shape[1]-self.__kernelSize+1))

        for channel in range(nChannels):
            for i in range(0, output.shape[1]):
                for j in range(0, output.shape[2]):
                    window = inputImg[i:i+self.__kernelSize,
                                      j:j+self.__kernelSize, channel]
                    value = self.operator(window)
                    output[channel][i][j] = value

        output = output.transpose((1, 2, 0))
        output = np.clip(output, 0, 255)
        print(f"outputImgSize= {output.shape}")
        return output

    def grayScaleTransform(self, img: np.ndarray) -> np.ndarray:
        grayImg = Functions.grayScaleTransform(img)
        grayImg = np.expand_dims(grayImg, axis=-1)
        return grayImg

    def normalizeGrayScale(self, img: np.ndarray) -> np.ndarray:
        img = img.squeeze()

        min = np.min(img)
        max = np.max(img)
        img = (img - min)*255/(max-min)
        return img


class AveragePooling:
    def __init__(self,
                 kernelSize: int,
                 stride: int,
                 padding: int = 0) -> None:
        """Average Pooling

        Args:
            kernelSize (int): カーネルのサイズ
            stride (int): ストライド
            padding (int, optional): パディング. Defaults to 0.
        """
        self.avgPool = torch.nn.AvgPool2d(kernel_size=kernelSize,
                                          stride=stride,
                                          padding=padding)

    def apply(self, img: np.ndarray):
        imgTensor = torch.tensor(img, dtype=torch.float32)

        imgTensor = imgTensor.permute(2, 0, 1).unsqueeze(0)
        output: torch.Tensor = self.avgPool(imgTensor)
        output = output.squeeze().permute(1, 2, 0)
        output = output.numpy()
        return output


class MaxPooling:
    def __init__(self,
                 kernelSize: int,
                 stride: int,
                 padding: int = 0) -> None:
        """Max Pooling

        Args:
            kernelSize (int): カーネルのサイズ
            stride (int): ストライド
            padding (int, optional): パディング. Defaults to 0.
        """
        self.maxPool = torch.nn.MaxPool2d(kernel_size=kernelSize,
                                          stride=stride,
                                          padding=padding)

    def apply(self, img: np.ndarray):
        imgTensor = torch.tensor(img, dtype=torch.float32)

        imgTensor = imgTensor.permute(2, 0, 1).unsqueeze(0)
        output: torch.Tensor = self.maxPool(imgTensor)
        output = output.squeeze().permute(1, 2, 0)
        output = output.numpy()
        return output


class GaussianFilter(Convolution):
    def __init__(self,
                 sigma: float,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Gaussian Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

        self.__filter: np.ndarray = self.__initFilter(sigma, kernelSize)

    def __initFilter(self, sigma: float, kernelSize: int):
        kernel = np.zeros([kernelSize, kernelSize])
        for iy in range(kernelSize):
            for ix in range(kernelSize):
                kernel[iy, ix] = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(- (
                    (ix - kernelSize // 2) ** 2 + (iy - kernelSize // 2) ** 2) / (2 * sigma ** 2))

        kernel /= kernel.sum()
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output


class MedianFilter(Convolution):
    def __init__(self,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Median Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

    def operator(self, input: np.ndarray):
        output = np.median(input)
        return output


class SmoothingFilter(Convolution):
    def __init__(self,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Smoothing Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

    def operator(self, input: np.ndarray):
        output = np.average(input)
        return output


class MotionFilter(Convolution):
    def __init__(self,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Motion Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

        self.__filter: np.ndarray = self.__initFilter(kernelSize)

    def __initFilter(self, kernelSize: int):
        diag = np.ones(shape=kernelSize) / kernelSize
        kernel = np.diag(diag)
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output


class MaxMinFilter(Convolution):
    def __init__(self,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Max-Min Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

    def operator(self, input: np.ndarray):
        min = np.min(input)
        max = np.max(input)

        return max - min

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class DifferentialFilter(Convolution):
    def __init__(self,
                 direction: str,
                 padding: int = 0) -> None:
        """Differential Filter

        Args:
            direction (str): 微分の方向, "x" or "y"
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=3,
                         padding=padding)
        self.__filter: np.ndarray = self.__initFilter(direction)

    def __initFilter(self, direction: str):
        kernel = None

        if direction == "x":
            kernel = np.array([[0, 0, 0],
                               [-1, 1, 0],
                               [0, 0, 0]])
        elif direction == "y":
            kernel = np.array([[0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class PrewittFilter(Convolution):
    def __init__(self,
                 direction: str,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Prewitt Filter

        Args:
            direction (str): 微分の方向, "x" or "y"
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)
        self.__filter: np.ndarray = self.__initFilter(direction, kernelSize)

    def __initFilter(self, direction: str, kernelSize: int):
        kernel = np.zeros(shape=(kernelSize, kernelSize))

        if direction == "x":
            for i in range(kernelSize):
                kernel[i][0] = 1
                kernel[i][-1] = -1
        elif direction == "y":
            for i in range(kernelSize):
                kernel[0][i] = 1
                kernel[-1][i] = -1
        else:
            kernel = None

        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class SobelFilter(Convolution):
    def __init__(self,
                 direction: str,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """Sobel Filter

        Args:
            direction (str): 微分の方向, "x" or "y"
            kernelSize (int): カーネルのサイズ(奇数)
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

        if kernelSize % 2 == 0:
            raise ValueError(f"KernelSize must be odd, not {kernelSize}")

        self.__filter: np.ndarray = self.__initFilter(direction, kernelSize)

    def __initFilter(self, direction: str, kernelSize: int):
        kernel = np.zeros(shape=(kernelSize, kernelSize))

        if direction == "x":
            for i in range(kernelSize):
                kernel[i][0] = 1
                kernel[i][-1] = -1
            kernel[kernelSize//2][0] = 2
            kernel[kernelSize//2][-1] = -2
        elif direction == "y":
            for i in range(kernelSize):
                kernel[0][i] = 1
                kernel[-1][i] = -1
            kernel[0][kernelSize//2] = 2
            kernel[-1][kernelSize//2] = -2
        else:
            kernel = None

        print(f"kernel= {kernel}")
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class LaplacianFilter(Convolution):
    def __init__(self,
                 padding: int = 0) -> None:
        """Laplacian Filter

        Args:
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=3,
                         padding=padding)
        self.__filter: np.ndarray = self.__initFilter()

    def __initFilter(self):
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

        print(f"kernel= {kernel}")
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class EmbossFilter(Convolution):
    def __init__(self,
                 padding: int = 0) -> None:
        """Emboss Filter

        Args:
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=3,
                         padding=padding)
        self.__filter: np.ndarray = self.__initFilter()

    def __initFilter(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])

        print(f"kernel= {kernel}")
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output

    def apply(self, img: np.ndarray) -> np.ndarray:
        grayImg = self.grayScaleTransform(img)
        appliedImg = super().apply(grayImg)
        appliedImg = self.normalizeGrayScale(appliedImg)
        return appliedImg


class LoGFilter(Convolution):
    def __init__(self,
                 sigma: float,
                 kernelSize: int,
                 padding: int = 0) -> None:
        """LoG Filter

        Args:
            kernelSize (int): カーネルのサイズ
            padding (int, optional): パディング. Defaults to 0.
        """
        super().__init__(kernelSize=kernelSize,
                         padding=padding)

        self.__filter: np.ndarray = self.__initFilter(sigma, kernelSize)

    def __initFilter(self, sigma: float, kernelSize: int):
        kernel = np.zeros((kernelSize, kernelSize), dtype=np.float32)
        pad_x = kernelSize // 2
        pad_y = kernelSize // 2
        for x in range(-pad_y, -pad_y + kernelSize):
            for y in range(-pad_x, -pad_x + kernelSize):
                kernel[y + pad_y, x + pad_y] = (x ** 2 + y ** 2 - 2 * (
                    sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        kernel /= (2 * np.pi * (sigma ** 6))
        kernel /= kernel.sum()
        return kernel

    def operator(self, input: np.ndarray):
        output = input * self.__filter
        output = np.sum(output)
        return output
