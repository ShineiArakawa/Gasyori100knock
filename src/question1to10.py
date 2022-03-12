from functions import Functions
from convolution import AveragePooling, MaxPooling, GaussianFilter, MedianFilter

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
        # =================================================================================

        # Array ===========================================================================
        self.imgArray_128 = np.array(self.img_128)
        self.imgArray_256 = np.array(self.img_256)
        self.imgArray_512 = np.array(self.img_512)

        self.imgArray_256_noise = np.array(self.img_256_noise)
        # =================================================================================
        pass

    def showImage256(self):
        self.img_256.show("imori")

    def question1(self):
        """RGB画像をBGRに入れ替える
        """
        imgArray_bgr = self.img_256[:, :, ::-1]

        self.__plotImages([self.imgArray_256, imgArray_bgr])

    def question2(self):
        """グレースケール変換
        """
        imgArray_grayScale = Functions.grayScaleTransform(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_grayScale])

    def question3(self):
        """二値化
        """
        imgArray_grayScale = Functions.grayScaleTransform(self.imgArray_256)
        imgArray_binary = Functions.binaryTransform(
            self.imgArray_256, threshold=128)

        self.__plotImages(
            [self.imgArray_256, imgArray_grayScale, imgArray_binary])

    def question4(self):
        """大津の二値化
        """
        imgArray_grayScale = Functions.grayScaleTransform(self.imgArray_256)
        imgArray_binary = Functions.otsuBinaryTransform(self.imgArray_256)

        self.__plotImages(
            [self.imgArray_256, imgArray_grayScale, imgArray_binary])

    def question5(self):
        """HSV変換
        """
        hsv = Functions.rgb2hsv(self.imgArray_256)
        hsv[..., 0] = (hsv[..., 0] + 180) % 360
        imgArray_hsv = Functions.hsv2rgb(hsv)

        self.__plotImages([self.imgArray_256, imgArray_hsv])

    def question6(self):
        """減色
        """
        imgArray_subtracted = Functions.colorSubtraction(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_subtracted])

    def question7(self):
        """Average-Pooling
        """
        imgArray_Average = AveragePooling(
            kernelSize=8, stride=8, padding=0).apply(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_Average])

    def question8(self):
        """Max-Pooling
        """
        imgArray_Max = MaxPooling(
            kernelSize=8, stride=8, padding=0).apply(self.imgArray_256)

        self.__plotImages([self.imgArray_256, imgArray_Max])

    def question9(self):
        """Gaussian filter
        """
        sigma = 2.0
        kernelSize = 10

        imgArray_gaussian1 = GaussianFilter(
            sigma=sigma, kernelSize=kernelSize, padding=1).apply(self.imgArray_256_noise)
        imgArray_gaussian2 = GaussianFilter(
            sigma=sigma, kernelSize=kernelSize, padding=1).apply(imgArray_gaussian1)
        imgArray_gaussian3 = GaussianFilter(
            sigma=sigma, kernelSize=kernelSize, padding=1).apply(imgArray_gaussian2)

        self.__plotImages([self.imgArray_256_noise, imgArray_gaussian1,
                          imgArray_gaussian2, imgArray_gaussian3])

    def question10(self):
        """Median filter
        """

        imgArray_median = MedianFilter(
            kernelSize=5, padding=0).apply(self.imgArray_256_noise)

        self.__plotImages([self.imgArray_256_noise, imgArray_median])

    def __plotImages(self, imgs: List[np.ndarray]):
        nImgs = len(imgs)
        fig, ax = plt.subplots(1, nImgs, dpi=200)
        for i in range(nImgs):
            imgArray = imgs[i].astype(np.uint8)
            img = Image.fromarray(imgArray)
            if imgArray.ndim == 2:
                ax[i].imshow(img, cmap="gray")
            else:
                ax[i].imshow(img)

        plt.tight_layout()
        plt.show()
        plt.close()
        plt.clf()
        del fig
        del ax


def main():
    mainObject = Main()

    # mainObject.showImage256()
    # mainObject.question1()
    # mainObject.question2()
    # mainObject.question3()
    # mainObject.question4()
    # mainObject.question5()
    # mainObject.question6()
    # mainObject.question7()
    mainObject.question8()
    # mainObject.question9()
    # mainObject.question10()


if __name__ == "__main__":
    main()
