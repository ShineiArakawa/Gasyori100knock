from functions import Functions
from convolution import DifferentialFilter, EmbossFilter, LaplacianFilter, LoGFilter, MaxMinFilter, MotionFilter, PrewittFilter, SmoothingFilter, SobelFilter

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


class Main:
    def __init__(self) -> None:
        # 画像 ============================================================================
        self.img_128 = Image.open("../dataset/images/imori_128x128.png")
        self.img_256 = Image.open("../dataset/images/imori_256x256.png")
        self.img_512 = Image.open("../dataset/images/imori_512x512.png")

        self.img_256_noise = Image.open(
            "../dataset/images/imori_256x256_noise.png")
        self.img_256_dark = Image.open(
            "../dataset/images/imori_256x256_dark.png")
        # =================================================================================

        # Array ===========================================================================
        self.imgArray_128 = np.array(self.img_128)
        self.imgArray_256 = np.array(self.img_256)
        self.imgArray_512 = np.array(self.img_512)

        self.imgArray_256_noise = np.array(self.img_256_noise)
        self.imgArray_256_dark = np.array(self.img_256_dark)
        # =================================================================================
        pass

    def question11(self):
        """平滑化フィルタ
        """
        imgArray_smoothed = SmoothingFilter(
            kernelSize=9, padding=0).apply(self.imgArray_256_noise)

        self.__plotImages([self.imgArray_256_noise, imgArray_smoothed])

    def question12(self):
        """モーションフィルタ
        """
        imgArray_motion = MotionFilter(
            kernelSize=5, padding=0).apply(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_motion])

    def question13(self):
        """Max-Minフィルタ
        """
        imgArray_maxMin = MaxMinFilter(
            kernelSize=5, padding=0).apply(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_maxMin])

    def question14(self):
        """Differential Filter
        """
        imgArray_differentialX = DifferentialFilter(
            direction="x", padding=0).apply(self.imgArray_256)
        imgArray_differentialY = DifferentialFilter(
            direction="y", padding=0).apply(self.imgArray_256)

        self.__plotImages(
            [self.imgArray_256, imgArray_differentialX, imgArray_differentialY])

    def question15(self):
        """Prewitt Filter
        """
        imgArray_prewittX = PrewittFilter(
            direction="x", kernelSize=3, padding=0).apply(self.imgArray_256)
        imgArray_prewittY = PrewittFilter(
            direction="y", kernelSize=3, padding=0).apply(self.imgArray_256)

        self.__plotImages(
            [self.imgArray_256, imgArray_prewittX, imgArray_prewittY])

    def question16(self):
        """Sobel Filter
        """
        imgArray_sobelX = SobelFilter(
            direction="x", kernelSize=3, padding=0).apply(self.imgArray_256)
        imgArray_sobelY = SobelFilter(
            direction="y", kernelSize=3, padding=0).apply(self.imgArray_256)

        self.__plotImages(
            [self.imgArray_256, imgArray_sobelX, imgArray_sobelY])

    def question17(self):
        """Laplacian Filter
        """
        imgArray_laplacian = LaplacianFilter(
            padding=0).apply(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_laplacian])

    def question18(self):
        """Emboss Filter
        """
        imgArray_emboss = EmbossFilter(
            padding=1).apply(self.imgArray_256)
        imgArray_emboss -= imgArray_emboss.min()  # normalize > [0, 1]
        imgArray_emboss /= imgArray_emboss.max()
        self.__plotImages([self.imgArray_256, imgArray_emboss])

    def question19(self):
        """LoG filter
        """
        sigma = 10.0
        kernelSize = 5

        imgArray_LoG = LoGFilter(sigma=sigma,
                                 kernelSize=kernelSize,
                                 padding=1).apply(self.imgArray_256_noise)

        self.__plotImages([self.imgArray_256_noise, imgArray_LoG])

    def question20(self):
        """Histgram
        """

        img_flatten = np.ravel(self.img_256_dark)
        plt.hist(img_flatten, bins=256, rwidth=0.8, range=(0, 255))
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

    # mainObject.question11()
    # mainObject.question12()
    # mainObject.question13()
    # mainObject.question14()
    # mainObject.question15()
    # mainObject.question16()
    # mainObject.question17()
    # mainObject.question18()
    # mainObject.question19()
    mainObject.question20()


if __name__ == "__main__":
    main()
