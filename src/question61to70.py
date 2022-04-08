"""https://github.com/yoyoyo-yo/Gasyori100knock
"""

import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from functions import Functions


class Main:
    def __init__(self) -> None:
        # 画像 ============================================================================
        self.img_128 = Image.open(
            "../dataset/images/imori_128x128.png"
        )
        self.img_256 = Image.open(
            "../dataset/images/imori_256x256.png"
        )
        self.img_512 = Image.open(
            "../dataset/images/imori_512x512.png"
        )
        self.img_256_noise = Image.open(
            "../dataset/images/imori_256x256_noise.png"
        )
        self.img_256_dark = Image.open(
            "../dataset/images/imori_256x256_dark.png"
        )
        self.img_256_gamma = Image.open(
            "../dataset/images/imori_256x256_gamma.png"
        )
        self.img_thorino = Image.open(
            "../dataset/images/thorino.jpg"
        )
        self.img_eye = Image.open(
            "../dataset/images/imori_256x256_eye.png"
        ).convert("RGB")
        self.img_seg = Image.open(
            "../dataset/images/seg_sample.png"
        )
        self.img_connect = Image.open(
            "../dataset/images/connect_sample.png"
        )
        self.img_gazo = Image.open(
            "../dataset/images/gazo_sample.png"
        )
        # =================================================================================

        # Array ===========================================================================
        self.imgArray_128 = np.array(self.img_128,
                                     dtype=np.float64)
        self.imgArray_256 = np.array(self.img_256,
                                     dtype=np.float64)
        self.imgArray_256_alreadyGrayScaled = np.array(self.img_256.convert("L"),
                                                       dtype=np.float64)
        self.imgArray_512 = np.array(self.img_512,
                                     dtype=np.float64)
        self.imgArray_512_alreadyGrayScaled = np.array(self.img_512.convert("L"),
                                                       dtype=np.float64)
        self.imgArray_256_noise = np.array(self.img_256_noise,
                                           dtype=np.float64)
        self.imgArray_256_dark = np.array(self.img_256_dark,
                                          dtype=np.float64)
        self.imgArray_256_gamma = np.array(self.img_256_gamma,
                                           dtype=np.float64)
        self.imgArray_thorino = np.array(self.img_thorino,
                                         dtype=np.float64)
        self.imgArray_eye = np.array(self.img_eye,
                                     dtype=np.float64)
        self.imgArray_seg = np.array(self.img_seg,
                                     dtype=np.float64)
        self.imgArray_connect = np.array(self.img_connect,
                                         dtype=np.float64)
        self.imgArray_gazo = np.array(self.img_gazo,
                                      dtype=np.float64)
        # =================================================================================
        pass

    def question61(self):
        """4-連結数
        """
        out = Functions.connect_4(self.imgArray_connect)

        self.__plotImages([self.imgArray_connect, out])

    def question62(self):
        """"8-連結数
        """
        out = Functions.connect_8(self.imgArray_connect)

        self.__plotImages([self.imgArray_connect, out])

    def question63(self):
        """細線化処理
        """
        out = Functions.thinning(self.imgArray_gazo)

        self.__plotImages([self.imgArray_gazo, out])

    def question64(self):
        """ヒルディッチの細線化
        """
        out = Functions.hilditchThinning(self.imgArray_gazo)
        self.__plotImages([self.imgArray_gazo, out])

    def question65(self):
        """Zhang-Suenの細線化
        """
        out = Functions.zhangSuenThinning(self.imgArray_gazo)
        self.__plotImages([self.imgArray_gazo, out])

    def question66(self):
        """HOG (Step.1) 勾配強度・勾配角度
        """
        # 1. gray -> gradient x and y
        gx, gy = Functions.getGradXY(self.imgArray_512_alreadyGrayScaled)

        # 2. get gradient magnitude and angle
        mag, grad = Functions.getMagAndGrad(gx, gy)

        # 3. quantization
        grad_q = Functions.histgramQuantization(grad)

        self.__plotImages([self.imgArray_512_alreadyGrayScaled, gx, gy, mag, grad_q])

    def question67(self):
        """HOG (Step.2) 勾配ヒストグラム
        """
        # 1. gray -> gradient x and y
        gx, gy = Functions.getGradXY(self.imgArray_512_alreadyGrayScaled)

        # 2. get gradient magnitude and angle
        mag, grad = Functions.getMagAndGrad(gx, gy)

        # 3. quantization
        grad_q = Functions.histgramQuantization(grad)
        histogram = Functions.gradientHistogram(grad_q, mag)
        self.__plotImages([histogram[..., i] for i in range(9)])

    def question68(self):
        """HOG (Step.3) ヒストグラム正規化
        """
        
        gx, gy = Functions.getGradXY(self.imgArray_512_alreadyGrayScaled)
        mag, grad = Functions.getMagAndGrad(gx, gy)
        grad_q = Functions.histgramQuantization(grad)
        histogram = Functions.gradientHistogram(grad_q, mag)
        histogram_norm = Functions.hogNormalization(histogram)

        fig, ax = plt.subplots(2, 9, figsize=(16, 4))
        for i in range(9):
            ax[0, i].set_title(f"hist {i}")
            ax[0, i].imshow(histogram[..., i])
            ax[1, i].set_title(f"hist norm {i}")
            ax[1, i].imshow(histogram_norm[..., i])
        plt.show()

    def question69(self):
        """HOG (Step.4) 特徴量の描画
        """
        # draw HOG
        gx, gy = Functions.getGradXY(self.imgArray_512_alreadyGrayScaled)
        mag, grad = Functions.getMagAndGrad(gx, gy)
        grad_q = Functions.histgramQuantization(grad)
        histogram = Functions.gradientHistogram(grad_q, mag)
        histogram_norm = Functions.hogNormalization(histogram)
        out = Functions.drawHOG(self.imgArray_512_alreadyGrayScaled, histogram_norm)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title("input")
        ax[0].imshow(self.imgArray_512_alreadyGrayScaled, cmap="gray")
        ax[1].set_title("output")
        ax[1].imshow(out, cmap="gray")
        plt.show()

    def question70(self):
        """カラートラッキング???
        """
        # hsv = Functions.rgb2hsv(self.imgArray_512)
        hsv = np.array(self.img_512.convert("HSV"), dtype=np.float64)
        mask = Functions.getMask(hsv)

        img = self.imgArray_512.astype(np.int8)
        mask = np.clip(mask, 0, 255).astype(np.int8)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].set_title("input")
        ax[0].imshow(img)
        ax[1].set_title("mask")
        ax[1].imshow(mask, cmap="gray")
        plt.show()

    def __plotImages(self, imgs: List[np.ndarray]):
        nImgs = len(imgs)
        fig, ax = plt.subplots(1, nImgs, dpi=200)
        for i in range(nImgs):
            imgArray = imgs[i].astype(np.uint8)
            if imgArray.ndim == 2:
                ax[i].imshow(imgArray, cmap="gray")
            else:
                img = Image.fromarray(imgArray)
                ax[i].imshow(img)

        plt.tight_layout()
        plt.show()
        plt.close()
        plt.clf()
        del fig
        del ax


def main():
    mainObject = Main()

    # mainObject.question61()
    # mainObject.question62()
    # mainObject.question63()
    # mainObject.question64()
    # mainObject.question65()
    # mainObject.question66()
    # mainObject.question67()
    # mainObject.question68()
    # mainObject.question69()
    mainObject.question70()


if __name__ == "__main__":
    main()
