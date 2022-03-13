from asyncio import windows_events
from turtle import shape
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


class Functions:
    @staticmethod
    def grayScaleTransform(imgArray: np.ndarray) -> np.ndarray:
        """RGB画像をグレースケール画像に変換する関数である

        Args:
            imgArray (np.ndarray): RGBの配列

        Returns:
            grayScale (np.ndarry): 2D ndarray
        """
        imgArray = imgArray.astype(np.float32)
        grayScale = 0.2126*imgArray[..., 0]+0.7152 * \
            imgArray[..., 1]+0.0722*imgArray[..., 2]
        grayScale = np.clip(grayScale, 0, 255)
        return grayScale

    @staticmethod
    def binaryTransform(imgArray: np.ndarray, threshold: int = 128):
        grayScale = Functions.grayScaleTransform(imgArray)
        grayScale[grayScale < threshold] = 0
        grayScale[grayScale >= threshold] = 255
        return grayScale

    @staticmethod
    def otsuBinaryTransform(imgArray: np.ndarray):
        grayScale = Functions.grayScaleTransform(imgArray)

        nThresholds = 256
        averageTotal = np.average(grayScale)
        variance = np.empty(shape=nThresholds)

        for threshold in range(0, nThresholds):
            class1_IDs = np.where(grayScale < threshold)
            class2_IDs = np.where(grayScale >= threshold)

            class1_nPixels = len(class1_IDs[0])
            class2_nPixels = len(class2_IDs[0])
            class1_avg = 0.0
            class2_avg = 0.0

            for i in range(class1_nPixels):
                class1_avg += grayScale[class1_IDs[0][i]][class1_IDs[1][i]]
            if class1_nPixels != 0:
                class1_avg /= class1_nPixels

            for i in range(class2_nPixels):
                class2_avg += grayScale[class2_IDs[0][i]][class2_IDs[1][i]]
            if class2_nPixels != 0:
                class2_avg /= class2_nPixels

            var = class1_nPixels*class2_nPixels*(class1_avg - class2_avg)**2
            var /= (class1_nPixels+class2_nPixels)**2

            variance[threshold] = var

        threshold = np.argmax(variance)
        print(f"threshold= {threshold}")
        grayScale[grayScale < threshold] = 0
        grayScale[grayScale >= threshold] = 255

        return grayScale

    @staticmethod
    def rgb2hsv(img: np.ndarray):
        """RGBからHSVに変化するためのメソッドである。
        Answerの内容を丸々コピーした。

        Args:
            img (np.ndarray): RGB

        Returns:
            np.ndarray: HSV
        """
        _img = img.copy().astype(np.float32)  # / 255
        v_max = _img.max(axis=2)
        v_min = _img.min(axis=2)
        v_argmin = _img.argmin(axis=2)
        hsv = np.zeros_like(_img, dtype=np.float32)
        r, g, b = np.split(_img, 3, axis=2)
        r, g, b = r[..., 0], g[..., 0], b[..., 0]

        diff = np.maximum(v_max - v_min, 1e-10)

        # Hue
        ind = v_argmin == 2
        hsv[..., 0][ind] = 60 * (g - r)[ind] / diff[ind] + 60
        ind = v_argmin == 0
        hsv[..., 0][ind] = 60 * (b - g)[ind] / diff[ind] + 180
        ind = v_argmin == 1
        hsv[..., 0][ind] = 60 * (r - b)[ind] / diff[ind] + 300
        ind = v_max == v_min
        hsv[..., 0][ind] = 0
        # Saturation
        hsv[..., 1] = v_max - v_min
        # Value
        hsv[..., 2] = v_max
        return hsv

    @staticmethod
    def hsv2rgb(hsv: np.ndarray):
        """RGBからHSVに変化するためのメソッドである。
        Answerの内容を丸々コピーした。

        Args:
            hsv (np.ndarray): HSV

        Returns:
            np.ndarray: RGB
        """
        h, s, v = np.split(hsv, 3, axis=2)
        h, s, v = h[..., 0], s[..., 0], v[..., 0]
        _h = h / 60
        x = s * (1 - np.abs(_h % 2 - 1))
        z = np.zeros_like(x)
        vals = np.array([[s, x, z], [x, s, z], [z, s, x],
                        [z, x, s], [x, z, s], [s, z, x]])

        img = np.zeros_like(hsv)

        for i in range(6):
            ind = _h.astype(int) == i
            for j in range(3):
                img[..., j][ind] = (v - s)[ind] + vals[i, j][ind]

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def colorSubtraction(img: np.ndarray, div=4):
        """減色を行うメソッドである。
        Answerの内容を丸々コピーした。

        Args:
            img (np.ndarray): 入力画像
            div (int, optional): 色の数。 Defaults to 4.

        Returns:
            np.ndarray: 減色された画像
        """
        th = 256 // div
        return np.clip(img // th * th + th // 2, 0, 255)

    @staticmethod
    def normalizeHistgram(img: np.ndarray, range: Tuple[int]):
        img = img.astype(np.float64)
        
        c = np.min(img)
        d = np.max(img)
        
        output = (range[1]-range[0])*(img-c)/(d-c)+range[0]
        output[img<c] = range[0]
        output[img>=d] = range[1]
        
        output = np.clip(output, 0, 255)
        
        return output
    
    @staticmethod
    def scaleShiftHistgram(img: np.ndarray, loc: float, scale: float):
        img = img.astype(np.float64)
        
        mean = np.mean(img)
        std = np.std(img)
        
        output = scale * (img - mean) / std + loc
        output = np.clip(output, 0, 255)
        
        return output
        
        