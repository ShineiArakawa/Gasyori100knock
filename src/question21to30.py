

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from functions import Functions


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
        self.imgArray_128 = np.array(self.img_128, dtype=np.float64)
        self.imgArray_256 = np.array(self.img_256, dtype=np.float64)
        self.imgArray_512 = np.array(self.img_512, dtype=np.float64)

        self.imgArray_256_noise = np.array(self.img_256_noise, dtype=np.float64)
        self.imgArray_256_dark = np.array(self.img_256_dark, dtype=np.float64)
        # =================================================================================
        pass

    def question21(self):
        """ヒストグラム正規化
        """
        imgArray_normalized = Functions.normalizeHistgram(self.imgArray_256_dark, (0, 255))
        
        self.__plotImages([self.imgArray_256_dark, imgArray_normalized])
        


    def question22(self):
        """スケーリングとシフト
        """
        imgArray_normalized = Functions.scaleShiftHistgram(self.imgArray_256_dark, 128, 50)
        
        self.__plotImages([self.imgArray_256_dark, imgArray_normalized])


    def question23(self):
        """Max-Minフィルタ
        """


    def question24(self):
        """Differential Filter
        """

    def question25(self):
        """Prewitt Filter
        """


    def question26(self):
        """Sobel Filter
        """


    def question27(self):
        """Laplacian Filter
        """

    def question28(self):
        """Emboss Filter
        """


    def question29(self):
        """LoG filter
        """


    def question30(self):
        """Histgram
        """

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

    # mainObject.question21()
    mainObject.question22()
    # mainObject.question23()
    # mainObject.question24()
    # mainObject.question25()
    # mainObject.question26()
    # mainObject.question27()
    # mainObject.question28()
    # mainObject.question29()
    # mainObject.question30()


if __name__ == "__main__":
    main()
