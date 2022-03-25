

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
        self.img_256_gamma = Image.open(
            "../dataset/images/imori_256x256_gamma.png")
        # =================================================================================

        # Array ===========================================================================
        self.imgArray_128 = np.array(self.img_128, dtype=np.float64)
        self.imgArray_256 = np.array(self.img_256, dtype=np.float64)
        self.imgArray_512 = np.array(self.img_512, dtype=np.float64)

        self.imgArray_256_noise = np.array(
            self.img_256_noise, dtype=np.float64)
        self.imgArray_256_dark = np.array(self.img_256_dark, dtype=np.float64)
        self.imgArray_256_gamma = np.array(
            self.img_256_gamma, dtype=np.float64)
        # =================================================================================
        pass

    def question21(self):
        """ヒストグラム正規化
        """
        imgArray_normalized = Functions.normalizeHistgram(
            self.imgArray_256_dark, (0, 255))

        self.__plotImages([self.imgArray_256_dark, imgArray_normalized])

    def question22(self):
        """スケーリングとシフト
        """
        imgArray_normalized = Functions.scaleShiftHistgram(
            self.imgArray_256_dark, 128, 50)

        self.__plotImages([self.imgArray_256_dark, imgArray_normalized])

    def question23(self):
        """ヒストグラム平坦化
        """
        algorithm = 2
        imgArray_equalized = Functions.equalizeHistgram(
            self.imgArray_256_dark, algorithm=algorithm)

        self.__plotImages([self.imgArray_256_dark, imgArray_equalized])

    def question24(self):
        """ガンマ補正
        """
        imgArray_gammaCorrected = Functions.gammaCorrection(
            self.imgArray_256_gamma, c=1, g=2.2)
        self.__plotImages([self.imgArray_256_gamma, imgArray_gammaCorrected])

    def question25(self):
        """最近傍補間
        """
        imgArray_magnified = Functions.magnify_nn(
            self.imgArray_256, magX=1.5, magY=1.5)
        self.__plotImages([self.imgArray_256, imgArray_magnified])

    def question26(self):
        """バイリニア補間
        """
        imgArray_magnified = Functions.magnify_biLinear(
            self.imgArray_256, magX=1.5, magY=2.0)
        self.__plotImages([self.imgArray_256, imgArray_magnified])

    def question27(self):
        """バイキュービック補間
        """
        imgArray_magnified = Functions.magnify_biCubic(
            self.imgArray_256, magX=10.0, magY=10.0)
        self.__plotImages([self.imgArray_256, imgArray_magnified])

    def question28(self):
        """平行移動
        """
        h, w, c = self.imgArray_512.shape
        imgArray_transformed = Functions.affine(
            self.imgArray_512, [[1, 0, 30], [0, 1, -30]], (h, w))
        self.__plotImages([self.imgArray_512, imgArray_transformed])

    def question29(self):
        """拡大縮小
        """
        magX = 1.3
        magY = 0.8
        shiftX = 30
        shiftY = -30

        h, w, c = self.imgArray_512.shape
        imgArray_transformed = Functions.affine(self.imgArray_512,
                                                [[magX, 0, shiftX], [
                                                    0, magY, shiftY]],
                                                (int(h*magY), int(w*magX)))
        self.__plotImages([self.imgArray_512, imgArray_transformed])

    def question30(self):
        """回転
        """
        # (1) ==================================================================
        # h, w, c = self.imgArray_512.shape
        # rotationMat = Functions.getRotationMat(-30.0)
        # imgArray_transformed = Functions.affine(self.imgArray_512,
        #                                         rotationMat,
        #                                         (h, w))
        # self.__plotImages([self.imgArray_512, imgArray_transformed])
        
        # (2) ==================================================================
        # ======================================================================
        # 自前の実装ではうまくいかなかった
        # ======================================================================
        # h, w, c = self.imgArray_512.shape
        
        # centerX = w // 2
        # centerY = h // 2
        # print(f"centerX= {centerX}, centerY= {centerY}")
        
        # shiftMatPre = Functions.getShiftMat(-centerX, -centerY, True)
        # rotationMat = Functions.getRotationMat(-30.0, True)
        # shiftMatAfter = Functions.getShiftMat(centerX, centerY, True)
        # transMat = np.dot(np.dot(shiftMatPre, rotationMat), shiftMatAfter)
        # # transMat = np.matmul(shiftMatPre, rotationMat)
        # # transMat = shiftMatAfter
        # print(transMat)
        
        # imgArray_transformed = Functions.affine(self.imgArray_512,
        #                                         transMat[:2],
        #                                         (h, w))
        
        # ======================================================================
        # 以下はホームページからコピペした実装
        # ======================================================================
        rad = - 30 / 180 * np.pi
        h, w, c = self.imgArray_512.shape
        tx = int((np.cos(- rad) - 1) * w // 2 - np.sin(- rad) * h // 2)
        ty = int(np.sin(- rad) * w // 2 + (np.cos(- rad) - 1) * h // 2)

        imgArray_transformed = Functions.affine(self.imgArray_512, [[np.cos(rad), - np.sin(rad), tx], [np.sin(rad), np.cos(rad), ty]], (h, w))

        self.__plotImages([self.imgArray_512, imgArray_transformed])

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
    # mainObject.question22()
    # mainObject.question23()
    # mainObject.question24()
    # mainObject.question25()
    # mainObject.question26()
    # mainObject.question27()
    # mainObject.question28()
    # mainObject.question29()
    mainObject.question30()


if __name__ == "__main__":
    main()
