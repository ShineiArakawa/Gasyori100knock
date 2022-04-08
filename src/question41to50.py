

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
        self.img_128 = Image.open("../dataset/images/imori_128x128.png")
        self.img_256 = Image.open("../dataset/images/imori_256x256.png")
        self.img_512 = Image.open("../dataset/images/imori_512x512.png")

        self.img_256_noise = Image.open(
            "../dataset/images/imori_256x256_noise.png")
        self.img_256_dark = Image.open(
            "../dataset/images/imori_256x256_dark.png")
        self.img_256_gamma = Image.open(
            "../dataset/images/imori_256x256_gamma.png")
        
        self.img_thorino = Image.open(
            "../dataset/images/thorino.jpg"
        )
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
        self.imgArray_thorino = np.array(
            self.img_thorino, dtype=np.float64)
        # =================================================================================
        pass

    def question41(self):
        """Cannyエッジ検出 (Step.1) エッジ強度
        うまくいかない。
        """
        # gaussian filtering
        gray = Functions.grayScaleTransform(self.imgArray_256).astype(np.float32)
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        # get edge strength, angle
        edge, angle = Functions.getEdgeAngle(fx, fy)

        # angle quantization
        angle = Functions.angleQuantization(angle)

        self.__plotImages([gray,
                           edge,
                           angle])

    def question42(self):
        """"Cannyエッジ検出 (Step.2) 細線化
        うまくいかない。
        """
        # gaussian filtering
        gray = Functions.grayScaleTransform(self.imgArray_256).astype(np.float32)
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        # get edge strength, angle
        edge, angle = Functions.getEdgeAngle(fx, fy)

        # angle quantization
        angle = Functions.angleQuantization(angle)
        edge_nms = Functions.nonMaximumSuppression(angle, edge)
        
        self.__plotImages([gray,
                           edge,
                           edge_nms])

    def question43(self):
        """Cannyエッジ検出 (Step.3) ヒステリシス閾処理
        """
        # gaussian filtering
        gray = Functions.grayScaleTransform(self.imgArray_256).astype(np.float32)
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        # get edge strength, angle
        edge, angle = Functions.getEdgeAngle(fx, fy)

        # angle quantization
        angle = Functions.angleQuantization(angle)
        edge_nms = Functions.nonMaximumSuppression(angle, edge)
        
        out = Functions.hysterisis(edge_nms, 50, 20)
        
        self.__plotImages([gray,
                           out])

    def question44(self):
        """Hough変換・直線検出 (Step.1) Hough変換
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_thorino)
        gaussian = cv2.GaussianBlur(grayScale, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        edge, angle = Functions.getEdgeAngle(fx, fy)
        angle = Functions.angleQuantization(angle)
        edge_nms = Functions.nonMaximumSuppression(angle, edge)
        canny = Functions.hysterisis(edge_nms, 50, 20)

        vote = Functions.voting(canny)
        
        self.__plotImages([grayScale, vote])

    def question45(self):
        """Hough変換・直線検出 (Step.2) NMS
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_thorino)
        gaussian = cv2.GaussianBlur(grayScale, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        edge, angle = Functions.getEdgeAngle(fx, fy)
        angle = Functions.angleQuantization(angle)
        edge_nms = Functions.nonMaximumSuppression(angle, edge)
        canny = Functions.hysterisis(edge_nms, 50, 20)

        vote = Functions.voting(canny)
        
        vote_nms = Functions.nonMaximumSuppressionHoughLines(vote)
        self.__plotImages([grayScale, vote_nms])

    def question46(self):
        """Hough変換・直線検出 (Step.3) Hough逆変換
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_thorino)
        gaussian = cv2.GaussianBlur(grayScale, (5, 5), 1.4)
        fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

        edge, angle = Functions.getEdgeAngle(fx, fy)
        angle = Functions.angleQuantization(angle)
        edge_nms = Functions.nonMaximumSuppression(angle, edge)
        canny = Functions.hysterisis(edge_nms, 50, 20)

        vote = Functions.voting(canny)
        
        vote_nms = Functions.nonMaximumSuppressionHoughLines(vote)
        out = Functions.inverseHough(vote_nms, self.imgArray_thorino)
        self.__plotImages([grayScale, out])

    def question47(self):
        """モルフォロジー処理(膨張)
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        out = Functions.morphologyErode(otsu, repeat=2)

        self.__plotImages([self.imgArray_256, otsu, out])
        
        
        
    def question48(self):
        """モルフォロジー処理(収縮)
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        out = Functions.morphologyDilate(otsu, repeat=2)

        self.__plotImages([self.imgArray_256, otsu, out])

    def question49(self):
        """オープニング処理
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        time = 1
        out = Functions.morphologyDilate(otsu, repeat=time)
        out = Functions.morphologyErode(out, repeat=time)
        
        self.__plotImages([self.imgArray_256, otsu, out])

    def question50(self):
        """クロージング処理
        """
        otsu = Functions.otsuBinaryTransform(self.imgArray_256)
        time = 1
        out = Functions.morphologyErode(otsu, repeat=time)
        out = Functions.morphologyDilate(out, repeat=time)
        
        self.__plotImages([self.imgArray_256, otsu, out])
        

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

    # mainObject.question41()
    # mainObject.question42()
    # mainObject.question43()
    # mainObject.question44()
    # mainObject.question45()
    # mainObject.question46()
    # mainObject.question47()
    # mainObject.question48()
    # mainObject.question49()
    mainObject.question50()


if __name__ == "__main__":
    main()
