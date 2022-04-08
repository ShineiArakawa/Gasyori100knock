

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
        # =================================================================================

        # Array ===========================================================================
        self.imgArray_128 = np.array(self.img_128,
                                     dtype=np.float64)
        self.imgArray_256 = np.array(self.img_256,
                                     dtype=np.float64)
        self.imgArray_512 = np.array(self.img_512,
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
        # =================================================================================
        pass

    def question51(self):
        """モルフォロジー勾配
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        time = 1
        eroded = Functions.morphologyErode(otsu, repeat=time)
        dilated = Functions.morphologyDilate(otsu, repeat=time)

        out = np.abs(eroded - dilated) * 255

        self.__plotImages([self.imgArray_256, out])

    def question52(self):
        """"トップハット変換
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        time = 3
        out = Functions.morphologyDilate(otsu, repeat=time)
        out = Functions.morphologyErode(out, repeat=time)

        out = np.abs(otsu - out) * 255

        self.__plotImages([self.imgArray_256, out])

    def question53(self):
        """ブラックハット変換
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        time = 3
        out = Functions.morphologyErode(otsu, repeat=time)
        out = Functions.morphologyDilate(out, repeat=time)

        out = np.abs(otsu - out) * 255

        self.__plotImages([self.imgArray_256, out])

    def question54(self):
        """テンプレートマッチング SSD
        """
        Functions.templateMatchingSSD(self.imgArray_256, self.imgArray_eye)

    def question55(self):
        """テンプレートマッチング SAD
        """
        Functions.templateMatchingSAD(self.imgArray_256, self.imgArray_eye)


    def question56(self):
        """テンプレートマッチング NCC
        """
        Functions.templateMatchingNCC(self.imgArray_256, self.imgArray_eye)


    def question57(self):
        """テンプレートマッチング ZNCC
        """
        Functions.templateMatchingZNCC(self.imgArray_256, self.imgArray_eye)


    def question58(self):
        """ラベリング 4近傍
        """
        out = Functions.labeling_4nn(self.imgArray_seg)

        self.__plotImages([self.imgArray_seg, out])

    def question59(self):
        """ラベリング 8近傍
        """
        out = Functions.labeling_8nn(self.imgArray_seg)

        self.__plotImages([self.imgArray_seg, out])

    def question60(self):
        """クロージング処理
        """
        thorino = np.array(self.img_thorino.resize((256,256)), dtype=np.float64)
        out = Functions.alphaBlend(self.imgArray_256, thorino, 0.6)

        self.__plotImages([self.imgArray_256, thorino, out])

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

    # mainObject.question51()
    # mainObject.question52()
    # mainObject.question53()
    # mainObject.question54()
    # mainObject.question55()
    # mainObject.question56()
    # mainObject.question57()
    # mainObject.question58()
    # mainObject.question59()
    mainObject.question60()


if __name__ == "__main__":
    main()
