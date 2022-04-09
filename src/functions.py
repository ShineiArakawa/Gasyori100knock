# autopep8: off
import imp
import os
import time
if os.name == "nt":
    from asyncio import windows_events

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io

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
        _img = img.copy().astype(np.float64)  # / 255
        v_max = _img.max(axis=2)
        v_min = _img.min(axis=2)
        v_argmin = _img.argmin(axis=2)
        hsv = np.zeros_like(_img, dtype=np.float64)
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
        out_h, out_w = map(int, out_shape)
        # out_h, out_w = out_shape

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

    @staticmethod
    def DFT(imgArray: np.ndarray):
        import cupy as cp
        imgArray = cp.asarray(imgArray)

        height, width = imgArray.shape
        freq = cp.empty(shape=(height, width), dtype=cp.float32)
        xMesh, yMesh = cp.meshgrid(cp.arange(0, width),
                                   cp.arange(0, height))

        for i in range(height):
            for j in range(width):
                freq[i][j] = cp.sum(imgArray *
                                    cp.exp(-2j*cp.pi*(j*xMesh/width + i*yMesh/height)))

        freq = cp.asnumpy(freq)
        return freq

    @staticmethod
    def IDFT(freq: np.ndarray):
        import cupy as cp
        freq = cp.asarray(freq)

        height, width = freq.shape
        imgArray = cp.empty(shape=(height, width),
                            dtype=cp.float32)
        xMesh, yMesh = cp.meshgrid(cp.arange(0, width),
                                   cp.arange(0, height))

        for i in range(height):
            for j in range(width):
                imgArray[i][j] = cp.abs(
                    cp.sum(freq * cp.exp(2j*cp.pi*(j*xMesh/width + i*yMesh/height))))

        imgArray /= (height * width)
        imgArray = cp.asnumpy(imgArray)
        return imgArray

    @staticmethod
    def lowPassFilter(G, ratio=0.5):
        H, W = G.shape
        h_half = H // 2
        w_half = W // 2

        # transfer positions
        _G = np.zeros_like(G)
        _G[:h_half, :w_half] = G[h_half:, w_half:]
        _G[:h_half, w_half:] = G[h_half:, :w_half]
        _G[h_half:, :w_half] = G[:h_half, w_half:]
        _G[h_half:, w_half:] = G[:h_half, :w_half]

        # filtering
        x, y = np.meshgrid(np.arange(0, W) - w_half, np.arange(0, H) - h_half)

        r = np.sqrt(x ** 2 + y ** 2)
        mask = np.ones((H, W), dtype=np.float32)
        mask[r > (h_half * ratio)] = 0
        _G *= mask

        # reverse original positions
        out = np.zeros_like(_G)
        out[:h_half, :w_half] = _G[h_half:, w_half:]
        out[:h_half, w_half:] = _G[h_half:, :w_half]
        out[h_half:, :w_half] = _G[:h_half, w_half:]
        out[h_half:, w_half:] = _G[:h_half, :w_half]

        return out

    @staticmethod
    def highPassFilter(G, ratio=0.1):
        H, W = G.shape
        h_half = H // 2
        w_half = W // 2

        # transfer positions
        _G = np.zeros_like(G)
        _G[:h_half, :w_half] = G[h_half:, w_half:]
        _G[:h_half, w_half:] = G[h_half:, :w_half]
        _G[h_half:, :w_half] = G[:h_half, w_half:]
        _G[h_half:, w_half:] = G[:h_half, :w_half]

        # filtering
        x, y = np.meshgrid(np.arange(0, W) - w_half, np.arange(0, H) - h_half)
        r = np.sqrt(x ** 2 + y ** 2)
        mask = np.ones((H, W), dtype=np.float32)
        mask[r < (h_half * ratio)] = 0
        _G *= mask

        # reverse original positions
        out = np.zeros_like(_G)
        out[:h_half, :w_half] = _G[h_half:, w_half:]
        out[:h_half, w_half:] = _G[h_half:, :w_half]
        out[h_half:, :w_half] = _G[:h_half, w_half:]
        out[h_half:, w_half:] = _G[:h_half, :w_half]

        return out

    @staticmethod
    def bandPassFilter(G, low=0.1, high=0.5):
        H, W = G.shape
        h_half = H // 2
        w_half = W // 2

        # transfer positions
        _G = np.zeros_like(G)
        _G[:h_half, :w_half] = G[h_half:, w_half:]
        _G[:h_half, w_half:] = G[h_half:, :w_half]
        _G[h_half:, :w_half] = G[:h_half, w_half:]
        _G[h_half:, w_half:] = G[:h_half, :w_half]

        # filtering
        x, y = np.meshgrid(np.arange(0, W) - w_half, np.arange(0, H) - h_half)
        r = np.sqrt(x ** 2 + y ** 2)
        mask = np.ones((H, W), dtype=np.float32)
        mask[(r < (h_half * low)) | (r > (h_half * high))] = 0
        _G *= mask

        # reverse original positions
        out = np.zeros_like(_G)
        out[:h_half, :w_half] = _G[h_half:, w_half:]
        out[:h_half, w_half:] = _G[h_half:, :w_half]
        out[h_half:, :w_half] = _G[:h_half, w_half:]
        out[h_half:, w_half:] = _G[:h_half, :w_half]

        return out

    @staticmethod
    def DCT(img, T=8, channel=3, toEnableCupy=False):
        if toEnableCupy:
            try:
                import cupy as cp
                np = cp
                img = cp.asarray(img)
            except:
                toEnableCupy = False

        H, W, _ = img.shape
        F = np.zeros((H, W, channel), dtype=np.float32)

        theta = np.pi / (2 * T)

        for c in range(channel):
            for vi in range(0, H, T):
                for ui in range(0, W, T):
                    for v in range(T):
                        for u in range(T):
                            cu = 1 / np.sqrt(2) if u == 0 else 1
                            cv = 1 / np.sqrt(2) if v == 0 else 1
                            coef1, coef2 = np.meshgrid(np.cos(
                                (2 * np.arange(0, T) + 1) * u * theta), np.cos((2 * np.arange(0, T) + 1) * v * theta))
                            F[vi + v, ui + u, c] = 2 * cu * cv * \
                                np.sum(img[vi: vi + T, ui: ui + T, c]
                                       * coef1 * coef2) / T

        if toEnableCupy:
            F = cp.asnumpy(F)
        return F

    @staticmethod
    def IDCT(F,  T=8, K=8, channel=3, toEnableCupy=False):
        if toEnableCupy:
            try:
                import cupy as cp
                np = cp
                F = cp.asarray(F)
            except:
                toEnableCupy = False

        H, W, _ = F.shape
        out = np.zeros((H, W, channel), dtype=np.float32)

        theta = np.pi / (2 * T)

        c_mat = np.ones([T, T])
        c_mat[0] /= np.sqrt(2)
        c_mat[:, 0] /= np.sqrt(2)

        for c in range(channel):
            for yi in range(0, H, T):
                for xi in range(0, W, T):
                    for y in range(T):
                        for x in range(T):
                            coef1, coef2 = np.meshgrid(np.cos(
                                (2 * x + 1) * np.arange(0, T) * theta), np.cos((2 * y + 1) * np.arange(0, T) * theta))
                            out[yi + y, xi + x, c] = 2 * np.sum(
                                F[yi: yi + K, xi: xi + K, c] * coef1[:K, :K] * coef2[:K, :K] * c_mat[:K, :K]) / T

        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)

        if toEnableCupy:
            out = cp.asnumpy(out)

        return out

    # MSE
    def mse(img1, img2):
        h, w, c = img1.shape
        mse = np.sum((img1 - img2) ** 2) / (h * w * c)
        return mse

    # PSNR
    def PSNR(img1, img2, vmax=255):
        _mse = 1e-10 if Functions.mse(img1,
                                      img2) == 0 else Functions.mse(img1,
                                                                    img2)
        return 10 * np.log10(vmax * vmax / _mse)

    # bitrate
    def bitrate(t, k):
        return 1. * t * (k ** 2) / (t ** 2)

    def quantization(F, T=8):
        h, w, channel = F.shape

        Q = np.array([[12, 18, 24, 30, 36, 42, 48, 54],
                      [18, 24, 30, 36, 42, 48, 54, 60],
                      [24, 30, 36, 42, 48, 54, 60, 66],
                      [30, 36, 42, 48, 54, 60, 66, 72],
                      [36, 42, 48, 54, 60, 66, 72, 78],
                      [42, 48, 54, 60, 66, 72, 78, 84],
                      [48, 54, 60, 66, 72, 78, 84, 90],
                      [54, 60, 66, 72, 78, 84, 90, 96]])

        for ys in range(0, h, T):
            for xs in range(0, w, T):
                for c in range(channel):
                    F[ys: ys + T, xs: xs + T,
                        c] = np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q

        return F

    def rgb2ycbcr(img):
        h, w, _ = img.shape
        ycbcr = np.zeros([h, w, 3], dtype=np.float32)
        ycbcr[..., 0] = 0.2990 * img[..., 2] + \
            0.5870 * img[..., 1] + 0.1140 * img[..., 0]
        ycbcr[..., 1] = -0.1687 * img[..., 2] - 0.3313 * \
            img[..., 1] + 0.5 * img[..., 0] + 128.
        ycbcr[..., 2] = 0.5 * img[..., 2] - 0.4187 * \
            img[..., 1] - 0.0813 * img[..., 0] + 128.
        return ycbcr

    # Y Cb Cr -> BGR
    def ycbcr2rgb(ycbcr):
        h, w, _ = ycbcr.shape
        out = np.zeros([h, w, 3], dtype=np.float32)
        out[..., 2] = ycbcr[..., 0] + (ycbcr[..., 2] - 128.) * 1.4020
        out[..., 1] = ycbcr[..., 0] - \
            (ycbcr[..., 1] - 128.) * 0.3441 - (ycbcr[..., 2] - 128.) * 0.7139
        out[..., 0] = ycbcr[..., 0] + (ycbcr[..., 1] - 128.) * 1.7718

        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        return out

    def getEdgeAngle(fx, fy):
        edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
        fx = np.maximum(fx, 1e-5)
        angle = np.arctan(fy / fx)
        return edge, angle

    def angleQuantization(angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135
        return _angle

    def nonMaximumSuppression(angle: np.ndarray, edge: np.ndarray):
        print(angle.shape)
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                if x == 0:
                    dx1 = max(dx1, 0)
                    dx2 = max(dx2, 0)
                if x == W-1:
                    dx1 = min(dx1, 0)
                    dx2 = min(dx2, 0)
                if y == 0:
                    dy1 = max(dy1, 0)
                    dy2 = max(dy2, 0)
                if y == H-1:
                    dy1 = min(dy1, 0)
                    dy2 = min(dy2, 0)
                if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                    _edge[y, x] = 0

        return _edge

    def hysterisis(edge, HT=100, LT=30):
        H, W = edge.shape

        # Histeresis threshold
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
        _edge[1: H + 1, 1: W + 1] = edge

        # 8 - Nearest neighbor
        nn = np.array(((1., 1., 1.), (1., 0., 1.),
                      (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H+2):
            for x in range(1, W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0

        edge = _edge[1:H+1, 1:W+1]

        return edge

    def voting(edge):
        H, W = edge.shape
        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)

        # hough table
        hough = np.zeros((rho_max * 2, 180), dtype=int)

        # get index of edge
        ind = np.where(edge == 255)

        # hough transformation
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))
                # vote
                hough[rho + rho_max, theta] += 1

        out = hough.astype(np.uint8)
        return out

        # non maximum suppression
    def nonMaximumSuppressionHoughLines(hough):
        rho_max, _ = hough.shape

        # non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x-1, 0)
                x2 = min(x+2, 180)
                y1 = max(y-1, 0)
                y2 = min(y+2, rho_max-1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
                    pass
                    #hough[y,x] = 255
                else:
                    hough[y, x] = 0

        # for hough visualization
        # get top-10 x index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        # get y index
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180
        _hough = np.zeros_like(hough, dtype=int)
        _hough[rhos, thetas] = 255

        return _hough

    def inverseHough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180 - rho_max / 2

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [255, 0, 0]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [255, 0, 0]

        out = out.astype(np.uint8)

        return out

    def morphologyErode(img, repeat=1):
        h, w = img.shape
        out = img.copy()

        # kernel
        mf = np.array(((0, 1, 0),
                       (1, 0, 1),
                       (0, 1, 0)), dtype=int)

        # each erode
        for i in range(repeat):
            tmp = np.pad(out, (1, 1), 'edge')
            # erode
            for y in range(1, h + 1):
                for x in range(1, w + 1):
                    if np.sum(mf * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                        out[y-1, x-1] = 0

        return out

    def morphologyDilate(img, repeat=1):
        h, w = img.shape

        # kernel
        mf = np.array(((0, 1, 0),
                       (1, 0, 1),
                       (0, 1, 0)), dtype=int)

        # each dilate time
        out = img.copy()
        for i in range(repeat):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, h+1):
                for x in range(1, w+1):
                    if np.sum(mf * tmp[y-1:y+2, x-1:x+2]) >= 255:
                        out[y-1, x-1] = 255

        return out

    def templateMatchingSSD(img, template):
        h, w, c = img.shape
        ht, wt, ct = template.shape

        resx, resy = -1, -1
        v = 255 * h * w * c

        for y in range(h - ht):
            for x in range(w - wt):
                _v = np.sum((img[y: y + ht, x: x + wt] - template) ** 2)

                if _v < v:
                    v = _v
                    resx, resy = x, y

        fig, ax = plt.subplots()
        ax.imshow(img.astype(np.uint8))
        ax.add_patch(plt.Rectangle((resx, resy), wt, ht,
                     fill=False, edgecolor='red', linewidth=3.5))
        plt.show()

    def templateMatchingSAD(img, template):
        h, w, c = img.shape
        ht, wt, ct = template.shape

        resx, resy = -1, -1
        v = 255 * h * w * c

        for y in range(h - ht):
            for x in range(w - wt):
                _v = np.sum(np.abs(img[y: y + ht, x: x + wt] - template))

                if _v < v:
                    v = _v
                    resx, resy = x, y

        fig, ax = plt.subplots()
        ax.imshow(img.astype(np.uint8))
        ax.add_patch(plt.Rectangle((resx, resy), wt, ht,
                     fill=False, edgecolor='red', linewidth=3.5))
        plt.show()

    def templateMatchingNCC(img, template):
        h, w, c = img.shape
        ht, wt, ct = template.shape

        resx, resy = -1, -1
        v = -1

        for y in range(h - ht):
            for x in range(w - wt):
                _v = np.sum(img[y: y + ht, x: x + wt] * template)
                _v /= (np.sqrt(np.sum(img[y: y + ht, x: x + wt]
                       ** 2)) * np.sqrt(np.sum(template ** 2)))
                if _v > v:
                    v = _v
                    resx, resy = x, y

        fig, ax = plt.subplots()
        ax.imshow(img.astype(np.uint8))
        ax.add_patch(plt.Rectangle((resx, resy), wt, ht,
                     fill=False, edgecolor='red', linewidth=3.5))
        plt.show()

    def templateMatchingZNCC(img, template):
        h, w, c = img.shape
        ht, wt, ct = template.shape

        _img = img.copy() - img.mean()
        _template = template.copy() - template.mean()

        resx, resy = -1, -1
        v = -1

        for y in range(h - ht):
            for x in range(w - wt):
                _v = np.sum(_img[y: y + ht, x: x + wt] * template)
                _v /= (np.sqrt(np.sum(_img[y: y + ht, x: x + wt]
                       ** 2)) * np.sqrt(np.sum(_template ** 2)))
                if _v > v:
                    v = _v
                    resx, resy = x, y

        fig, ax = plt.subplots()
        ax.imshow(img.astype(np.uint8))
        ax.add_patch(plt.Rectangle((resx, resy), wt, ht,
                     fill=False, edgecolor='red', linewidth=3.5))
        plt.show()

    def labeling_4nn(img):
        h, w = img.shape

        label = np.zeros((h, w), dtype=int)
        label[img > 0] = 1

        # look up table
        LUT = [0 for _ in range(h * w)]

        n = 1

        for y in range(h):
            for x in range(w):
                # skip black pixel
                if label[y, x] == 0:
                    continue

                # get above pixel
                c3 = label[max(y-1, 0), x]

                # get left pixel
                c5 = label[y, max(x-1, 0)]

                # if not labeled
                if c3 < 2 and c5 < 2:
                    # labeling
                    n += 1
                    label[y, x] = n
                else:
                    # replace min label index
                    _vs = [c3, c5]
                    vs = [a for a in _vs if a > 1]
                    v = min(vs)
                    label[y, x] = v

                    minv = v
                    for _v in vs:
                        if LUT[_v] != 0:
                            minv = min(minv, LUT[_v])
                    for _v in vs:
                        LUT[_v] = minv

        count = 1

        # integrate index of look up table
        for l in range(2, n+1):
            flag = True
            for i in range(n+1):
                if LUT[i] == l:
                    if flag:
                        count += 1
                        flag = False
                    LUT[i] = count

        # draw color
        COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
        out = np.zeros((h, w, 3), dtype=np.uint8)

        for i, lut in enumerate(LUT[2:]):
            out[label == (i+2)] = COLORS[lut-2]

        return out

    def labeling_8nn(img):
        # get image shape
        h, w = img.shape

        # prepare labeling image
        label = np.zeros((h, w), dtype=int)
        label[img > 0] = 1

        # look up table
        LUT = [0 for _ in range(h * w)]

        n = 1

        for y in range(h):
            for x in range(w):
                if label[y, x] == 0:
                    continue
                # get right top pixel
                c2 = label[max(y-1, 0), min(x+1, w-1)]
                # get top pixel
                c3 = label[max(y-1, 0), x]
                # get left top pixel
                c4 = label[max(y-1, 0), max(x-1, 0)]
                # get left pixel
                c5 = label[y, max(x-1, 0)]

                # if all pixel is non labeled
                if c3 < 2 and c5 < 2 and c2 < 2 and c4 < 2:
                    n += 1
                    label[y, x] = n
                else:
                    # get labeled index
                    _vs = [c3, c5, c2, c4]
                    vs = [a for a in _vs if a > 1]
                    v = min(vs)
                    label[y, x] = v

                    minv = v
                    for _v in vs:
                        if LUT[_v] != 0:
                            minv = min(minv, LUT[_v])
                    for _v in vs:
                        LUT[_v] = minv

        count = 1

        # integrate labeled index of look up table
        for l in range(2, n+1):
            flag = True
            for i in range(n+1):
                if LUT[i] == l:
                    if flag:
                        count += 1
                        flag = False
                    LUT[i] = count

        # draw color
        COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
        out = np.zeros((h, w, 3), dtype=np.uint8)

        for i, lut in enumerate(LUT[2:]):
            out[label == (i+2)] = COLORS[lut-2]

        return out

    def alphaBlend(img1, img2, alpha):
        out = img1 * alpha + img2 * (1 - alpha)
        out = np.clip(out, 0, 255)
        return out

    def connect_4(img):
        # get shape
        h, w = img.shape

        # prepare temporary image
        tmp = np.zeros((h, w), dtype=int)

        # binarize
        tmp[img > 0] = 1

        # prepare out image
        out = np.zeros((h, w, 3), dtype=np.uint8)

        # each pixel
        for y in range(h):
            for x in range(w):
                if tmp[y, x] < 1:
                    continue

                S = 0
                S += (tmp[y, min(x + 1, w - 1)] - tmp[y, min(x + 1, w - 1)] *
                      tmp[max(y - 1, 0), min(x + 1, w - 1)] * tmp[max(y - 1, 0), x])
                S += (tmp[max(y - 1, 0), x] - tmp[max(y - 1, 0), x] *
                      tmp[max(y - 1, 0), max(x - 1, 0)] * tmp[y, max(x - 1, 0)])
                S += (tmp[y, max(x - 1, 0)] - tmp[y, max(x - 1, 0)] *
                      tmp[min(y + 1, h - 1), max(x - 1, 0)] * tmp[min(y + 1, h - 1), x])
                S += (tmp[min(y + 1, h - 1), x] - tmp[min(y + 1, h - 1), x] *
                      tmp[min(y + 1, h - 1), min(x + 1, w - 1)] * tmp[y, min(x + 1, w - 1)])

                if S == 0:
                    out[y, x] = [0, 0, 255]
                elif S == 1:
                    out[y, x] = [0, 255, 0]
                elif S == 2:
                    out[y, x] = [255, 0, 0]
                elif S == 3:
                    out[y, x] = [255, 255, 0]
                elif S == 4:
                    out[y, x] = [255, 0, 255]

        out = out.astype(np.uint8)

        return out

    def connect_8(img):
        # get shape
        h, w = img.shape

        # prepare temporary
        _tmp = np.zeros((h, w), dtype=int)

        # get binarize
        _tmp[img > 0] = 1

        # inverse for connect 8
        tmp = 1 - _tmp

        # prepare image
        out = np.zeros((h, w, 3), dtype=np.uint8)

        # each pixel
        for y in range(h):
            for x in range(w):
                if _tmp[y, x] < 1:
                    continue

                S = 0
                S += (tmp[y, min(x + 1, w - 1)] - tmp[y, min(x + 1, w - 1)] *
                      tmp[max(y - 1, 0), min(x + 1, w - 1)] * tmp[max(y - 1, 0), x])
                S += (tmp[max(y - 1, 0), x] - tmp[max(y - 1, 0), x] *
                      tmp[max(y - 1, 0), max(x - 1, 0)] * tmp[y, max(x - 1, 0)])
                S += (tmp[y, max(x - 1, 0)] - tmp[y, max(x - 1, 0)] *
                      tmp[min(y + 1, h - 1), max(x - 1, 0)] * tmp[min(y + 1, h - 1), x])
                S += (tmp[min(y + 1, h - 1), x] - tmp[min(y + 1, h - 1), x] *
                      tmp[min(y + 1, h - 1), min(x + 1, w - 1)] * tmp[y, min(x + 1, w - 1)])

                if S == 0:
                    out[y, x] = [0, 0, 255]
                elif S == 1:
                    out[y, x] = [0, 255, 0]
                elif S == 2:
                    out[y, x] = [255, 0, 0]
                elif S == 3:
                    out[y, x] = [255, 255, 0]
                elif S == 4:
                    out[y, x] = [255, 0, 255]

        out = out.astype(np.uint8)

        return out

    def thinning(img):
        # get shape
        h, w = img.shape

        # prepare out image
        out = np.zeros((h, w), dtype=int)
        out[img > 0] = 1

        count = 1
        while count > 0:
            count = 0
            tmp = out.copy()
            # each pixel ( rasta scan )
            for y in range(h):
                for x in range(w):
                    # skip black pixel
                    if out[y, x] < 1:
                        continue

                    # count satisfied conditions
                    judge = 0

                    # condition 1
                    if (tmp[y, min(x + 1, w - 1)] + tmp[max(y - 1, 0), x] + tmp[y, max(x - 1, 0)] + tmp[min(y + 1, h - 1), x]) < 4:
                        judge += 1

                    # condition 2
                    c = 0
                    c += (tmp[y, min(x + 1, w - 1)] - tmp[y, min(x + 1, w - 1)] *
                          tmp[max(y - 1, 0), min(x + 1,  w - 1)] * tmp[max(y - 1, 0), x])
                    c += (tmp[max(y - 1, 0), x] - tmp[max(y - 1, 0), x] *
                          tmp[max(y - 1, 0), max(x - 1, 0)] * tmp[y, max(x - 1, 0)])
                    c += (tmp[y, max(x - 1, 0)] - tmp[y, max(x - 1, 0)] *
                          tmp[min(y + 1, h - 1), max(x - 1, 0)] * tmp[min(y + 1, h - 1), x])
                    c += (tmp[min(y + 1, h - 1), x] - tmp[min(y + 1, h - 1), x] *
                          tmp[min(y + 1, h - 1), min(x + 1, w - 1)] * tmp[y, min(x + 1,  w - 1)])
                    if c == 1:
                        judge += 1

                    # x condition 3
                    if np.sum(tmp[max(y - 1, 0): min(y + 2, h), max(x - 1, 0): min(x + 2, w)]) >= 4:
                        judge += 1

                    # if all conditions are satisfied
                    if judge == 3:
                        out[y, x] = 0
                        count += 1

        out = out.astype(np.uint8) * 255

        return out

    def hilditchThinning(img):
        # get shape
        H, W = img.shape

        # prepare out image
        out = np.zeros((H, W), dtype=int)
        out[img > 0] = 1

        # inverse pixel value
        tmp = out.copy()
        _tmp = 1 - tmp

        count = 1
        while count > 0:
            count = 0
            tmp = out.copy()
            _tmp = 1 - tmp

            tmp2 = out.copy()
            _tmp2 = 1 - tmp2

            # each pixel
            for y in range(H):
                for x in range(W):
                    # skip black pixel
                    if out[y, x] < 1:
                        continue

                    judge = 0

                    # condition 1
                    if (tmp[y, min(x+1, W-1)] * tmp[max(y-1, 0), x] * tmp[y, max(x-1, 0)] * tmp[min(y+1, H-1), x]) == 0:
                        judge += 1

                    # condition 2
                    c = 0
                    c += (_tmp[y, min(x+1, W-1)] - _tmp[y, min(x+1, W-1)] *
                          _tmp[max(y-1, 0), min(x+1, W-1)] * _tmp[max(y-1, 0), x])
                    c += (_tmp[max(y-1, 0), x] - _tmp[max(y-1, 0), x] *
                          _tmp[max(y-1, 0), max(x-1, 0)] * _tmp[y, max(x-1, 0)])
                    c += (_tmp[y, max(x-1, 0)] - _tmp[y, max(x-1, 0)] *
                          _tmp[min(y+1, H-1), max(x-1, 0)] * _tmp[min(y+1, H-1), x])
                    c += (_tmp[min(y+1, H-1), x] - _tmp[min(y+1, H-1), x] *
                          _tmp[min(y+1, H-1), min(x+1, W-1)] * _tmp[y, min(x+1, W-1)])
                    if c == 1:
                        judge += 1

                    # condition 3
                    if np.sum(tmp[max(y-1, 0): min(y+2, H), max(x-1, 0): min(x+2, W)]) >= 3:
                        judge += 1

                    # condition 4
                    if np.sum(out[max(y-1, 0): min(y+2, H), max(x-1, 0): min(x+2, W)]) >= 2:
                        judge += 1

                    # condition 5
                    _tmp2 = 1 - out

                    c = 0
                    c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] *
                          _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                    c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] *
                          (1 - tmp[max(y-1, 0), max(x-1, 0)]) * _tmp2[y, max(x-1, 0)])
                    c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] *
                          _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                    c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] *
                          _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                    if c == 1 or (out[max(y-1, 0), max(x-1, 0)] != tmp[max(y-1, 0), max(x-1, 0)]):
                        judge += 1

                    c = 0
                    c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] *
                          _tmp2[max(y-1, 0), min(x+1, W-1)] * (1 - tmp[max(y-1, 0), x]))
                    c += ((1-tmp[max(y-1, 0), x]) - (1 - tmp[max(y-1, 0), x])
                          * _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                    c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] *
                          _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                    c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] *
                          _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                    if c == 1 or (out[max(y-1, 0), x] != tmp[max(y-1, 0), x]):
                        judge += 1

                    c = 0
                    c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] *
                          (1 - tmp[max(y-1, 0), min(x+1, W-1)]) * _tmp2[max(y-1, 0), x])
                    c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] *
                          _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                    c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] *
                          _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                    c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] *
                          _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                    if c == 1 or (out[max(y-1, 0), min(x+1, W-1)] != tmp[max(y-1, 0), min(x+1, W-1)]):
                        judge += 1

                    c = 0
                    c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] *
                          _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                    c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] *
                          _tmp2[max(y-1, 0), max(x-1, 0)] * (1 - tmp[y, max(x-1, 0)]))
                    c += ((1 - tmp[y, max(x-1, 0)]) - (1 - tmp[y, max(x-1, 0)]) *
                          _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                    c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] *
                          _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                    if c == 1 or (out[y, max(x-1, 0)] != tmp[y, max(x-1, 0)]):
                        judge += 1

                    if judge >= 8:
                        out[y, x] = 0
                        count += 1

        out = out.astype(np.uint8) * 255

        return out

    def zhangSuenThinning(img):
        # get shape
        H, W = img.shape

        # prepare out image
        out = np.zeros((H, W), dtype=int)
        out[img > 0] = 1

        # inverse
        out = 1 - out

        while True:
            s1 = []
            s2 = []

            # step 1 ( rasta scan )
            for y in range(1, H-1):
                for x in range(1, W-1):

                    # condition 1
                    if out[y, x] > 0:
                        continue

                    # condition 2
                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1

                    if f1 != 1:
                        continue

                    # condition 3
                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue

                    # condition 4
                    if out[y-1, x] + out[y, x+1] + out[y+1, x] < 1:
                        continue

                    # condition 5
                    if out[y, x+1] + out[y+1, x] + out[y, x-1] < 1:
                        continue

                    s1.append([y, x])

            for v in s1:
                out[v[0], v[1]] = 1

            # step 2 ( rasta scan )
            for y in range(1, H-1):
                for x in range(1, W-1):

                    # condition 1
                    if out[y, x] > 0:
                        continue

                    # condition 2
                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1

                    if f1 != 1:
                        continue

                    # condition 3
                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue

                    # condition 4
                    if out[y-1, x] + out[y, x+1] + out[y, x-1] < 1:
                        continue

                    # condition 5
                    if out[y-1, x] + out[y+1, x] + out[y, x-1] < 1:
                        continue

                    s2.append([y, x])

            for v in s2:
                out[v[0], v[1]] = 1

            # if not any pixel is changed
            if len(s1) < 1 and len(s2) < 1:
                break

        out = 1 - out
        out = out.astype(np.uint8) * 255

        return out

    # Magnitude and gradient
    def getGradXY(gray):
        h, w = gray.shape

        # padding before grad
        _gray = np.pad(gray, (1, 1), 'edge').astype(float)

        # get grad x
        gx = _gray[1: h + 1, 2:] - _gray[1: h + 1, : w]
        # get grad y
        gy = _gray[2:, 1: w + 1] - _gray[: h, 1: w + 1]
        # replace 0 with
        gx[np.abs(gx) <= 1e-10] = 1e-10
        return gx, gy

    # get magnitude and gradient
    def getMagAndGrad(gx, gy):
        # get gradient maginitude
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        grad = np.arctan(gy / gx)
        grad[grad < 0] = np.pi / 2 + grad[grad < 0] + np.pi / 2

        return mag, grad

    # gradient histogram
    def histgramQuantization(grad):
        # prepare quantization table
        grad_q = np.zeros_like(grad, dtype=float)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            grad_q[np.where((grad >= d * i) & (grad <= d * (i + 1)))] = i

        return grad_q

    def gradientHistogram(grad_q, mag, n=8):
        h, w = mag.shape

        # get cell num
        cell_n_h = h // n
        cell_n_w = w // n
        histogram = np.zeros((cell_n_h, cell_n_w, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_n_h):
            for x in range(cell_n_w):
                for j in range(n):
                    for i in range(n):
                        histogram[y, x, int(
                            grad_q[y * n + j, x * n + i])] += mag[y * n + j, x * n + i]

        return histogram

    def hogNormalization(histogram, c=3, epsilon=1):
        cell_n_h, cell_n_w, hist_c = histogram.shape
        hist_norm = histogram.copy()

        # each histogram
        for y in range(cell_n_h):
            for x in range(cell_n_w):
                for c in range(hist_c):
                    hist_norm[y, x, c] /= np.sqrt(np.sum(histogram[max(y - 1, 0): min(
                        y + 2, cell_n_h), max(x - 1, 0): min(x + 2, cell_n_w), c] ** 2) + epsilon)

        return hist_norm

    def drawHOG(gray, histogram, n=8):
        h, w = gray.shape
        cell_n_h, cell_n_w, _ = histogram.shape

        out = gray[1:  h + 1, 1:  w + 1].copy().astype(np.uint8)
        out = (out * 0.8).astype(np.uint8)

        for y in range(cell_n_h):
            for x in range(cell_n_w):
                cx = x * n + n // 2
                cy = y * n + n // 2
                x1 = cx + n // 2 - 1
                y1 = cy
                x2 = cx - n // 2 + 1
                y2 = cy

                h = histogram[y, x] / np.sum(histogram[y, x])
                h /= h.max()

                for c in range(9):
                    #angle = (20 * c + 10 - 90) / 180. * np.pi
                    # get angle
                    angle = (20 * c + 10) / 180. * np.pi
                    rx = int(np.sin(angle) * (x1 - cx) +
                             np.cos(angle) * (y1 - cy) + cx)
                    ry = int(np.cos(angle) * (x1 - cx) -
                             np.cos(angle) * (y1 - cy) + cy)
                    lx = int(np.sin(angle) * (x2 - cx) +
                             np.cos(angle) * (y2 - cy) + cx)
                    ly = int(np.cos(angle) * (x2 - cx) -
                             np.cos(angle) * (y2 - cy) + cy)

                    # color is HOG value
                    c = int(255. * h[c])

                    # draw line
                    cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)

        return out

    def getMask(hsv) -> np.ndarray:
        h = hsv[..., 0]
        mask = np.zeros_like(h).astype(np.uint8)
        mask[((h < 90) | (h > 140))] = 1
        return mask

    def morphologyOpen(img, repeat=1):
        out = Functions.morphologyDilate(img, repeat=repeat)
        out = Functions.morphologyErode(out, repeat=repeat)
        return out

    def morphologyClose(img, repeat=1):
        out = Functions.morphologyErode(img, repeat=repeat)
        out = Functions.morphologyDilate(out, repeat=repeat)
        return out

    def bilinearInterGray(img, a, b):
        h, w = img.shape
        out_h = int(h * a)
        out_w = int(w * b)

        xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

        _xs = np.floor(xs / b).astype(int)  # original x
        _ys = np.floor(ys / a).astype(int)  # original y

        dx = xs / b - _xs
        dy = ys / a - _ys

        # dx = np.repeat(np.expand_dims(dx, axis=-1), c, axis=-1) # repeat channel
        # dy = np.repeat(np.expand_dims(dy, axis=-1), c, axis=-1) # repeat channel

        _xs1p = np.minimum(_xs + 1, w - 1)
        _ys1p = np.minimum(_ys + 1, h - 1)

        out = (1 - dx) * (1 - dy) * img[_ys, _xs] + dx * (1 - dy) * img[_ys, _xs1p] + \
            (1 - dx) * dy * img[_ys1p, _xs] + dx * dy * img[_ys1p, _xs1p]

        return np.clip(out, 0, 255).astype(np.uint8)

    def gaborFilter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
        # get half size
        d = K_size // 2

        # prepare kernel
        gabor = np.zeros((K_size, K_size), dtype=np.float32)

        # each value
        for y in range(K_size):
            for x in range(K_size):
                # distance from center
                px = x - d
                py = y - d

                # degree -> radian
                theta = angle / 180. * np.pi

                # get kernel x
                _x = np.cos(theta) * px + np.sin(theta) * py

                # get kernel y
                _y = -np.sin(theta) * px + np.cos(theta) * py

                # fill kernel
                gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) /
                                     (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

        # kernel normalization
        gabor /= np.sum(np.abs(gabor))

        return gabor

    def gaborFiltering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
        H, W = gray.shape
        # padding
        gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

        # prepare out image
        out = np.zeros((H, W), dtype=np.float32)

        # get gabor filter
        gabor = Functions.gaborFilter(
            K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

        # filtering
        for y in range(H):
            for x in range(W):
                out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        return out

    def hessianCorner(gray):
        # Sobel
        def sobelFiltering(gray):
            # get shape
            H, W = gray.shape

            # sobel kernel
            sobely = np.array(((1, 2, 1),
                               (0, 0, 0),
                               (-1, -2, -1)), dtype=np.float32)

            sobelx = np.array(((1, 0, -1),
                               (2, 0, -2),
                               (1, 0, -1)), dtype=np.float32)

            # padding
            tmp = np.pad(gray, (1, 1), 'edge')

            # prepare
            Ix = np.zeros_like(gray, dtype=np.float32)
            Iy = np.zeros_like(gray, dtype=np.float32)

            # get differential
            for y in range(H):
                for x in range(W):
                    Ix[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * sobelx)
                    Iy[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * sobely)

            Ix2 = Ix ** 2
            Iy2 = Iy ** 2
            Ixy = Ix * Iy

            return Ix2, Iy2, Ixy

        # Hessian
        def cornerDetect(gray, Ix2, Iy2, Ixy, pad=2):
            # get shape
            H, W = gray.shape

            # prepare for show detection
            out = np.array((gray, gray, gray))
            out = np.transpose(out, (1, 2, 0))

            # get Hessian value
            Hes = np.zeros((H, W))

            for y in range(H):
                for x in range(W):
                    Hes[y, x] = Ix2[y, x] * Iy2[y, x] - Ixy[y, x] ** 2

            # Detect Corner and show
            for y in range(H):
                for x in range(W):
                    if Hes[y, x] == np.max(Hes[max(y-1, 0): min(y+2, H), max(x-1, 0): min(x+2, W)]) and Hes[y, x] > np.max(Hes) * 0.1:
                        out[y - pad: y + pad, x - pad: x + pad] = [255, 0, 0]

            out = out.astype(np.uint8)

            return out

        #  image sobel
        Ix2, Iy2, Ixy = sobelFiltering(gray)

        # corner detection
        out = cornerDetect(gray, Ix2, Iy2, Ixy)

        return out

    def sobelFiltering(gray):
        # get shape
        H, W = gray.shape

        # sobel kernel
        sobely = np.array(((1, 2, 1),
                           (0, 0, 0),
                           (-1, -2, -1)), dtype=np.float32)

        sobelx = np.array(((1, 0, -1),
                           (2, 0, -2),
                           (1, 0, -1)), dtype=np.float32)

        # padding
        tmp = np.pad(gray, (1, 1), 'edge')

        # prepare
        Ix = np.zeros_like(gray, dtype=np.float32)
        Iy = np.zeros_like(gray, dtype=np.float32)

        # get differential
        for y in range(H):
            for x in range(W):
                Ix[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * sobelx)
                Iy[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * sobely)

        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        return Ix2, Iy2, Ixy

    # gaussian filtering

    def gaussianFiltering(I, K_size=3, sigma=3):
        # get shape
        H, W = I.shape

        # gaussian
        I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

        # gaussian kernel
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(K_size):
            for y in range(K_size):
                _x = x - K_size // 2
                _y = y - K_size // 2
                K[y, x] = np.exp(-(_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
        K /= (sigma * np.sqrt(2 * np.pi))
        K /= K.sum()

        # filtering
        for y in range(H):
            for x in range(W):
                I[y, x] = np.sum(I_t[y: y + K_size, x: x + K_size] * K)

        return I

    def cornerDetect(gray, Ix2, Iy2, Ixy, k=0.04, th=0.1):
        # prepare output image
        out = np.array((gray, gray, gray))
        out = np.transpose(out, (1, 2, 0))

        # get R
        R = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)

        # detect corner
        out[R >= np.max(R) * th] = [255, 0, 0]

        out = out.astype(np.uint8)

        return out

    def dicColor(img):
        img = img // 64 * 64 + 32
        return img

    def getFeature(img):
        feat = np.zeros(12, dtype=np.float32)

        for i in range(4):
            feat[i, ] = (img[..., 0] == (64 * i + 32)).sum()
            feat[i + 4] = (img[..., 1] == (64 * i + 32)).sum()
            feat[i + 8] = (img[..., 2] == (64 * i + 32)).sum()

        return feat

    def getDB():
        train_paths = [
            "../dataset/train/akahara_1.jpg",
            "../dataset/train/akahara_2.jpg",
            "../dataset/train/akahara_3.jpg",
            "../dataset/train/madara_1.jpg",
            "../dataset/train/madara_2.jpg",
            "../dataset/train/madara_3.jpg"
        ]

        # prepare database
        db = np.zeros((len(train_paths), 13), dtype=np.float32)

        # each image
        for i, path in enumerate(train_paths):
            print(path)
            img = io.imread(path)
            img = cv2.resize(
                img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = Functions.dicColor(img)
            feat = Functions.getFeature(img)
            db[i, :-1] = feat

            # get class
            if 'akahara' in path:
                cls = 0
            elif 'madara' in path:
                cls = 1

            # store class label
            db[i, -1] = cls

        return db, train_paths

    def assignLabelInit(db, paths, class_n=2):
        feats = db.copy()

        # assign random label
        np.random.seed(0)
        feats[:, -1] = np.random.randint(0, class_n, (len(db)))

        # prepare gravity
        gs = np.zeros((class_n, feats.shape[1] - 1), dtype=np.float32)

        # get gravity
        for i in range(class_n):
            gs[i] = np.mean(
                feats[np.where(feats[..., -1] == i)[0], :12], axis=0)

        print("Assigned label")
        print(feats[:, -1])
        print("Gravity")
        print(gs)

    def kmeans(db, paths, class_n=2):
        feats = db.copy()
        feat_n = feats.shape[1] - 1

        # assign random label
        np.random.seed(0)
        feats[:, -1] = np.random.randint(0, class_n, (len(db)))

        # prepare gravity
        gs = np.zeros((class_n, feat_n), dtype=np.float32)

        # get gravity
        for i in range(class_n):
            gs[i] = np.mean(
                feats[np.where(feats[..., -1] == i)[0], :feat_n], axis=0)

        while True:
            # prepare greavity
            gs = np.zeros((class_n, feat_n), dtype=np.float32)
            change_count = 0

            # compute gravity
            for i in range(class_n):
                gs[i] = np.mean(
                    feats[np.where(feats[..., -1] == i)[0], :feat_n], axis=0)

            # re-labeling
            for i in range(len(feats)):
                # get distance each nearest graviry
                dis = np.sqrt(
                    np.sum(np.square(np.abs(gs - feats[i, :feat_n])), axis=1))

                # get new label
                pred = np.argmin(dis, axis=0)

                # if label is difference from old label
                if int(feats[i, -1]) != pred:
                    change_count += 1
                    feats[i, -1] = pred

            if change_count < 1:
                break

        for i in range(db.shape[0]):
            print(paths[i].split("/")[-1], " Pred:", feats[i, -1])

    def getDBAll():
        train_paths = [
            "../dataset/train/akahara_1.jpg",
            "../dataset/train/akahara_2.jpg",
            "../dataset/train/akahara_3.jpg",
            "../dataset/train/madara_1.jpg",
            "../dataset/train/madara_2.jpg",
            "../dataset/train/madara_3.jpg",
            "../dataset/test/akahara_1.jpg",
            "../dataset/test/akahara_2.jpg",
            "../dataset/test/akahara_3.jpg",
            "../dataset/test/madara_1.jpg",
            "../dataset/test/madara_2.jpg",
            "../dataset/test/madara_3.jpg"
        ]
        
        # prepare database
        db = np.zeros((len(train_paths), 13), dtype=np.float32)

        # each image
        for i, path in enumerate(train_paths):
            print(path)
            img = io.imread(path)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = Functions.dicColor(img)
            feat = Functions.getFeature(img)
            db[i, :-1] = feat

            # get class
            if 'akahara' in path:
                cls = 0
            elif 'madara' in path:
                cls = 1

            # store class label
            db[i, -1] = cls

        return db, train_paths
    
    def colorKmeansStep1(img, class_n=5):
        #  get shape
        h, w, c = img.shape
        _img = np.reshape(img, (h * w, -1))

        np.random.seed(0)
        
        # select one index randomly
        i = np.random.choice(np.arange(h * w), class_n, replace=False)
        colors = _img[i].copy()

        print("random selected color")
        print(colors)

        # assign near color
        clss = np.zeros((h * w), dtype=int)

        for i in range(h * w):
            dis = np.sqrt(np.sum((colors - _img[i]) ** 2, axis=1))
            clss[i] = np.argmin(dis)

        out = clss.reshape(h, w)
        out = out.astype(np.uint8)
        return out
    
    def colorKmeans(img, class_n=5):
        h, w, c = img.shape
        _img = np.reshape(img, (h * w, -1))

        np.random.seed(0)

        # get index randomly
        i = np.random.choice(np.arange(h * w), class_n, replace=False)
        colors = _img[i].copy()

        while True:
            # prepare pixel class label
            clss = np.zeros((h * w), dtype=int)
            
            # each pixel
            for i in range(h * w):
                # get distance from index pixel
                dis = np.sqrt(np.sum((colors - _img[i])**2, axis=1))
                # get argmin distance
                clss[i] = np.argmin(dis)

            # selected pixel values
            colors_tmp = np.zeros((class_n, 3))
            
            # each class label
            for i in range(class_n):
                colors_tmp[i] = np.mean(_img[clss == i], axis=0)

            # if not any change
            if (colors == colors_tmp).all():
                break
            else:
                colors = colors_tmp.copy()

        # prepare out image
        out = np.zeros((h * w, 3), dtype=np.float32)

        # assign selected pixel values  
        for i in range(class_n):
            out[clss == i] = colors[i]

        print(colors)
            
        out = np.clip(out, 0, 255)
        out = np.reshape(out, (h, w, 3))
        out = out.astype(np.uint8)
        return out
    
    def getIOU(a, b):
        area_a = (a[2] - a[0]) * (a[3] - a[1]) # get area of a
        area_b = (b[2] - b[0]) * (b[3] - b[1]) # get area of b

        iou_x1 = np.maximum(a[0], b[0]) # get left top x of IoU
        iou_y1 = np.maximum(a[1], b[1]) # get left top y of IoU
        iou_x2 = np.minimum(a[2], b[2]) # get right bottom of IoU
        iou_y2 = np.minimum(a[3], b[3]) # get right bottom of IoU

        area_iou = np.maximum(iou_x2 - iou_x1, 0) * np.maximum(iou_y2 - iou_y1, 0) # get area of IoU
        iou = area_iou / (area_a + area_b - area_iou) # get overlap ratio between IoU and all area
        return iou
    
    def crop_bbox(img, gt, crop_n=200, crop_size=60, th=0.5):
        # get shape
        h, w, c = img.shape
        
        crops = []
        labels = []

        # each crop
        for i in range(crop_n):
            # get left top x of crop bounding box
            x1 = np.random.randint(w - crop_size)
            # get left top y of crop bounding box
            y1 = np.random.randint(h - crop_size)
            # get right bottom x of crop bounding box
            x2 = x1 + crop_size
            # get right bottom y of crop bounding box
            y2 = y1 + crop_size

            # crop bounding box
            crop = np.array((x1, y1, x2, y2))

            # get IoU between crop box and gt
            _iou = Functions.getIOU(gt, crop)
            
            crops.append(crop)

            # assign label
            if _iou >= th:
                labels.append(1)
            else:
                labels.append(0)
                
        return np.array(crops), np.array(labels)

    def hog(img):
        # Grayscale
        def rgb2gray(img):
            return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722

        # Magnitude and gradient
        def get_gradxy(gray):
            H, W = gray.shape

            # padding before grad
            gray = np.pad(gray, (1, 1), 'edge')

            # get grad x
            gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
            # get grad y
            gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
            # replace 0 with 
            gx[gx == 0] = 1e-6

            return gx, gy

        # get magnitude and gradient
        def get_maggrad(gx, gy):
            # get gradient maginitude
            magnitude = np.sqrt(gx ** 2 + gy ** 2)

            # get gradient angle
            gradient = np.arctan(gy / gx)

            gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

            return magnitude, gradient

        # Gradient histogram
        def quantization(gradient):
            # prepare quantization table
            gradient_quantized = np.zeros_like(gradient, dtype=int)

            # quantization base
            d = np.pi / 9

            # quantization
            for i in range(9):
                gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

            return gradient_quantized


        # get gradient histogram
        def gradient_histogram(gradient_quantized, magnitude, n=8):
            # get shape
            h, w = magnitude.shape

            # get cell num
            cell_n_h = h // n
            cell_n_w = w // n
            histogram = np.zeros((cell_n_h, cell_n_w, 9), dtype=np.float32)

            # each pixel
            for y in range(cell_n_h):
                for x in range(cell_n_w):
                    for j in range(n):
                        for i in range(n):
                            histogram[y, x, gradient_quantized[y * n + j, x * n + i]] += magnitude[y * n + j, x * n + i]

            return histogram

        # histogram normalization
        def normalization(histogram, c=3, epsilon=1):
            cell_n_h, cell_n_w, hist_c = histogram.shape
            ## each histogram
            for y in range(cell_n_h):
                for x in range(cell_n_w):
                    for c in range(hist_c):
                        histogram[y, x, c] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_n_h),  max(x - 1, 0) : min(x + 2, cell_n_w), c] ** 2) + epsilon)

            return histogram

        gray = rgb2gray(img)
        gx, gy = get_gradxy(gray)
        magnitude, gradient = get_maggrad(gx, gy)
        gradient_quantized = quantization(gradient)
        histogram = gradient_histogram(gradient_quantized, magnitude)
        histogram = normalization(histogram)
        return histogram


    # resize using bi-linear
    def resize(img, h, w):
        _h, _w, _c  = img.shape

        ah = 1. * h / _h
        aw = 1. * w / _w

        y = np.arange(h).repeat(w).reshape(w, -1)
        x = np.tile(np.arange(w), (h, 1))

        # get coordinate toward x and y of resized image
        y = (y / ah)
        x = (x / aw)

        # transfer to int
        ix = np.floor(x).astype(np.int32)
        iy = np.floor(y).astype(np.int32)

        # clip index
        ix = np.minimum(ix, _w-2)
        iy = np.minimum(iy, _h-2)

        # get distance between original image index and resized image index
        dx = x - ix
        dy = y - iy

        dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
        dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)
        
        out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
        out[out > 255] = 255

        return out


    # crop bounding box and make dataset
    def make_dataset(img, gt, crop_n=200, crop_size=60, th=0.5, size=32):
        h, w, _ = img.shape

        # get HOG feature dimension
        hog_feat_n = ((size // 8) ** 2) * 9

        db = np.zeros([crop_n, hog_feat_n + 1])

        # each crop
        for i in range(crop_n):
            x1 = np.random.randint(w - crop_size)
            y1 = np.random.randint(h - crop_size)
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            crop = np.array([x1, y1, x2, y2])
            crop_area = img[y1:y2, x1:x2]
            crop_area = Functions.resize(crop_area, size, size)

            _hog = Functions.hog(crop_area)
            db[i, :hog_feat_n] = _hog.ravel()
            db[i, -1] = 1 if Functions.getIOU(gt, crop) >= th else 0 # label

        return db
    
    def sliding_window_step1(img, size=32):
        h, w, _ = img.shape
        
        # base rectangle [h, w]
        recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

        # sliding window
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                for rec in recs:
                    # get half size of ractangle
                    dh = int(rec[0] // 2)
                    dw = int(rec[1] // 2)

                    x1 = max(x - dw, 0) # get left top x
                    x2 = min(x + dw, w) # get left top y
                    y1 = max(y - dh, 0) # get right bottom x
                    y2 = min(y + dh, h) # get right bottom y

                    # crop region
                    region = img[max(y - dh, 0) : min(y + dh, h), max(x - dw, 0) : min(x + dw, w)]

                    # resize crop region
                    region = Functions.resize(region, size, size)

                    # get HOG feature
                    region_hog = Functions.hog(region).ravel()
    
    def nms(bboxes, iou_th=0.5):
        _bboxes = bboxes.copy()
        res_inds = np.arange(len(_bboxes))
        selected_inds = []
        
        while len(res_inds) > 0:
            i = res_inds[np.argmax(_bboxes[res_inds, -1])]
            selected_inds.append(i)
            res_inds = np.array([x for x in res_inds if x != i], dtype=int)

            base_bb = _bboxes[i]
            res_bb = _bboxes[res_inds]
            
            _x1, _y1, _x2, _y2 = base_bb[:4]
            _s = np.maximum(_x2 - _x1 , 0) * np.maximum(_y2 - _y1, 0)

            res_s = np.maximum(res_bb[:, 2] - res_bb[:, 0], 0) * np.maximum(res_bb[:, 3] - res_bb[:, 1], 0)

            iou_x1 = np.maximum(_x1, res_bb[:, 0])
            iou_y1 = np.maximum(_y1, res_bb[:, 1])
            iou_x2 = np.minimum(_x2, res_bb[:, 2])
            iou_y2 = np.minimum(_y2, res_bb[:, 3])
            iou_s = np.maximum(iou_x2 - iou_x1, 0) * np.maximum(iou_y2 - iou_y1, 0)

            iou_ratio = iou_s / (res_s + _s - iou_s)
            
            delete_inds = res_inds[iou_ratio >= iou_th]
            res_inds = np.array([x for x in res_inds if x not in delete_inds])

        return selected_inds