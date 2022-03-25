# autopep8: off
import os
import time
if os.name == "nt":
    from asyncio import windows_events

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# autopep8: on


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
        output[img < c] = range[0]
        output[img >= d] = range[1]

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

    @staticmethod
    def equalizeHistgram(imgArray: np.ndarray, algorithm: int = 0) -> np.ndarray:
        """ヒストグラムの平坦化を行うメソッドである

        Args:
            imgArray (np.ndarray): 入力画像
            algorithm (int, optional): アルゴリズム番号. Defaults to 0.
                algorithm=0: グレースケール変化を施してからヒストグラムの平坦化を行う
                algorithm=1: R,G,Bそれぞれに対してヒストグラムの平坦化を行う
                algorithm=2: R,G,Bをまとめてヒストグラムの平坦化を行う

        Raises:
            ValueError: algorithmに不正な値が指定された場合に発生するエラーである

        Returns:
            np.ndarray: 出力画像
        """

        output = None

        if algorithm == 0:
            grayScale = Functions.grayScaleTransform(imgArray)
            output = np.zeros_like(grayScale)

            x_max = 255  # constant
            s = 1
            for dim in grayScale.shape:
                s *= dim
            coeff = x_max / s

            hist = np.histogram(grayScale.ravel(), bins=255, range=(0, 255))
            hist_cumsum = np.cumsum(hist[0])

            output = coeff * hist_cumsum[grayScale.astype(np.uint8)]

        elif algorithm == 1:
            output = np.zeros_like(imgArray)

            x_max = 255  # constant
            h, w, c = imgArray.shape
            s = h * w
            coeff = x_max / s

            for i in range(c):
                hist = np.histogram(
                    imgArray[:, :, i].ravel(), bins=255, range=(0, 255))
                hist_cumsum = np.cumsum(hist[0])

                output[:, :, i] = coeff * \
                    hist_cumsum[imgArray[:, :, i].astype(np.uint8)]

        elif algorithm == 2:
            output = np.zeros_like(imgArray)

            x_max = 255  # constant
            h, w, c = imgArray.shape
            s = h * w * c
            coeff = x_max / s

            hist = np.histogram(imgArray.ravel(), bins=255, range=(0, 255))
            hist_cumsum = np.cumsum(hist[0])

            output = coeff * hist_cumsum[imgArray.astype(np.uint8)]

        else:
            raise ValueError("Algorithm ID is invalid!!")

        output = np.clip(output, 0, 255)
        return output

    @staticmethod
    def gammaCorrection(imgArray: np.ndarray, c: float, g: float) -> np.ndarray:
        if imgArray.dtype is not np.float64:
            imgArray = imgArray.astype(np.float64)

        imgArray_normalized = imgArray / 255

        output = (imgArray_normalized / c)**(1/g) * 255

        output = np.clip(output, 0, 255)
        return output

    @staticmethod
    def magnify_nn(imgArray: np.ndarray, magX: float, magY: float) -> np.ndarray:
        outputHeight = int(imgArray.shape[0] * magX)
        outputWidth = int(imgArray.shape[1] * magY)

        nChannels = imgArray.shape[2]
        output = np.zeros(shape=(outputHeight, outputWidth,
                          nChannels), dtype=np.float64)

        meshX, meshY = np.meshgrid(range(outputHeight), range(outputWidth))
        output[meshY, meshX] = imgArray[np.round(
            meshY / magX).astype(int), np.round(meshX / magY).astype(int)]

        return output

    @staticmethod
    def magnify_biLinear(imgArray: np.ndarray, magX: float, magY: float) -> np.ndarray:
        # =============================================================
        # 自作で作成したプログラムであるが、うまく動作しない。
        # =============================================================
        # startTime = time.perf_counter()

        # height, width, nChannels = imgArray.shape
        # outputHeight = int(height * magX)
        # outputWidth = int(width * magY)
        # output = np.zeros(shape=(outputHeight, outputWidth,
        #                   nChannels), dtype=np.float64)

        # for channel in range(nChannels):
        #     for indexX in range(outputHeight):
        #         for indexY in range(outputWidth):
        #             x_origin = indexX/magY
        #             y_origin = indexY/magX

        #             indexX_origin = np.floor(x_origin).astype(int)
        #             indexY_origin = np.floor(y_origin).astype(int)

        #             dx = x_origin - indexX_origin
        #             dy = y_origin - indexY_origin

        #             indexX_origin_plus = np.minimum(
        #                 indexX_origin + 1, width - 1)
        #             indexY_origin_plus = np.minimum(
        #                 indexY_origin + 1, height - 1)

        #             pixel_0_0 = imgArray[indexY_origin][indexX_origin][channel]
        #             pixel_0_1 = imgArray[indexY_origin][indexX_origin_plus][channel]
        #             pixel_1_0 = imgArray[indexY_origin_plus][indexX_origin][channel]
        #             pixel_1_1 = imgArray[indexY_origin_plus][indexX_origin_plus][channel]

        #             output = (1 - dx) * (1 - dy) * pixel_0_0 + dx * (1 - dy) * pixel_0_1 + \
        #                 (1 - dx) * dy * pixel_1_0 + \
        #                 dx * dy * pixel_1_1

        # endTime = time.perf_counter()
        # print(f"elapsedTime= {endTime-startTime} [sec]")
        # return output

        # =============================================================
        # ホームページからコピーした実装である。
        # =============================================================
        img = imgArray
        a = magX
        b = magY

        h, w, c = img.shape
        out_h = int(h * a)
        out_w = int(w * b)

        xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

        _xs = np.floor(xs / b).astype(int)  # original x
        _ys = np.floor(ys / a).astype(int)  # original y

        dx = xs / b - _xs
        dy = ys / a - _ys

        dx = np.repeat(np.expand_dims(dx, axis=-1),
                       c, axis=-1)  # repeat channel
        dy = np.repeat(np.expand_dims(dy, axis=-1),
                       c, axis=-1)  # repeat channel

        _xs1p = np.minimum(_xs + 1, w - 1)
        _ys1p = np.minimum(_ys + 1, h - 1)

        out = (1 - dx) * (1 - dy) * img[_ys, _xs] + dx * (1 - dy) * img[_ys, _xs1p] + \
            (1 - dx) * dy * img[_ys1p, _xs] + dx * dy * img[_ys1p, _xs1p]

        return np.clip(out, 0, 255)

    @staticmethod
    def magnify_biCubic(imgArray: np.ndarray, magX: float, magY: float) -> np.ndarray:
        startTime = time.perf_counter()

        h, w, c = imgArray.shape
        out_h = int(h * magX)
        out_w = int(w * magY)
        output = np.zeros([out_h, out_w, c], dtype=np.float32)

        xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

        _xs = np.floor(xs / magY).astype(int)  # original x
        _ys = np.floor(ys / magX).astype(int)  # original y

        dx1 = np.abs(xs / magY - (_xs - 1))
        dx2 = np.abs(xs / magY - _xs)
        dx3 = np.abs(xs / magY - (_xs + 1))
        dx4 = np.abs(xs / magY - (_xs + 2))
        dy1 = np.abs(ys / magX - (_ys - 1))
        dy2 = np.abs(ys / magX - _ys)
        dy3 = np.abs(ys / magX - (_ys + 1))
        dy4 = np.abs(ys / magX - (_ys + 2))

        dxs = [dx1, dx2, dx3, dx4]
        dys = [dy1, dy2, dy3, dy4]

        def weight(t, a=-1):
            w = np.zeros_like(t)
            w[t <= 1] = ((a + 2) * (t ** 3) - (a + 3) * (t ** 2) + 1)[t <= 1]
            w[t > 1] = (a * (t ** 3) - 5 * a * (t ** 2) +
                        8 * a * t - 4 * a)[t > 1]
            return w

        w_sum = np.zeros_like(output, dtype=np.float32)

        for j in range(-1, 3):
            for i in range(-1, 3):
                ind_x = np.minimum(np.maximum(_xs + i, 0), w - 1)
                ind_y = np.minimum(np.maximum(_ys + j, 0), h - 1)

                wx = weight(dxs[i + 1])
                wy = weight(dys[j + 1])
                wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
                wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

                w_sum += wx * wy
                output += wx * wy * imgArray[ind_y, ind_x]

        output = np.clip(output, 0, 255)

        endTime = time.perf_counter()
        print(f"elapsedTime= {endTime-startTime} [sec]")

        return output

    @staticmethod
    def affine(img, affine_mat, out_shape):
        h, w, channel = img.shape

        [a, b, tx], [c, d, ty] = affine_mat
        out_h, out_w = out_shape

        out = np.zeros([out_h, out_w, channel])

        # pad for black
        img_pad = np.pad(img, [(1, 1), (1, 1), (0, 0)])

        xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

        _xs = np.maximum(np.minimum(
            (1 / (a * d - b * c) * (d * xs - b * ys) - tx).astype(int) + 1, w + 1), 0)
        _ys = np.maximum(np.minimum(
            (1 / (a * d - b * c) * (- c * xs + a * ys) - ty).astype(int) + 1, h + 1), 0)

        out[ys, xs] = img_pad[_ys, _xs]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def getShiftMat(x: int, y: int, toReturnAll: bool = False):
        if toReturnAll:
            return [[1, 0, x],
                    [0, 1, y],
                    [0, 0, 1]]
        return [[1, 0, x],
                [0, 1, y]]

    @staticmethod
    def getRotationMat(theta: float, toReturnAll: bool = False):
        theta = np.deg2rad(theta)
        if toReturnAll:
            return [[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]]
        return [[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0]]
