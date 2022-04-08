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

    def question71(self):
        """マスキング
        """
        hsv = cv2.cvtColor(self.imgArray_512.astype(
            np.uint8), cv2.COLOR_RGB2HSV)
        mask = Functions.getMask(hsv)

        img_masked = self.imgArray_512.astype(np.uint8).copy(
        ) * np.concatenate([mask[..., None], mask[..., None], mask[..., None]], axis=-1)

        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.imgArray_512.astype(np.uint8))
        ax[1].set_title("mask")
        ax[1].imshow(mask, cmap="gray")
        ax[2].set_title("mask")
        ax[2].imshow(img_masked, cmap="gray")
        plt.show()

    def question72(self):
        """"マスキング(カラートラッキング＋モルフォロジー)]
        うまくいかない。
        """
        hsv = cv2.cvtColor(self.imgArray_256.astype(
            np.uint8), cv2.COLOR_RGB2HSV)
        mask = Functions.getMask(hsv)

        mask2 = Functions.morphologyOpen(mask.astype(np.float64), repeat=10)
        mask2 = Functions.morphologyClose(mask2, repeat=10).astype(np.uint8)

        img_masked = self.imgArray_256.astype(np.uint8).copy(
        ) * np.concatenate([mask[..., None], mask[..., None], mask[..., None]], axis=-1)
        img_masked2 = self.imgArray_256.astype(np.uint8).copy(
        ) * np.concatenate([mask2[..., None], mask2[..., None], mask2[..., None]], axis=-1)

        fig, ax = plt.subplots(1, 5, figsize=(16, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.imgArray_256.astype(np.uint8))
        ax[1].set_title("mask before")
        ax[1].imshow(mask, cmap="gray")
        ax[2].set_title("masked before")
        ax[2].imshow(img_masked)
        ax[3].set_title("mask")
        ax[3].imshow(mask2, cmap="gray")
        ax[4].set_title("masked")
        ax[4].imshow(img_masked2)
        plt.show()

    def question73(self):
        """縮小と拡大
        """
        img_gray = np.array(self.img_256.convert("L"), dtype=np.float64)
        img_resized = Functions.bilinearInterGray(img_gray, a=0.5, b=0.5)
        img_resized = Functions.bilinearInterGray(img_resized, a=2, b=2)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(img_gray, cmap="gray")
        ax[1].set_title("rersized")
        ax[1].imshow(img_resized, cmap="gray")
        plt.show()

    def question74(self):
        """ピラミッド差分による高周波成分の抽出
        """
        img_gray = np.array(self.img_256.convert("L"))
        img_resized = Functions.bilinearInterGray(img_gray, a=0.5, b=0.5)
        img_resized = Functions.bilinearInterGray(img_resized, a=2, b=2)
        img_diff = np.abs(img_gray - img_resized)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(img_gray, cmap="gray")
        ax[1].set_title("diff")
        ax[1].imshow(img_diff, cmap="gray")
        plt.show()

    def question75(self):
        """ガウシアンピラミッド
        """
        img_gray = np.array(self.img_256.convert("L"))
        pyramid = [img_gray]

        fig, ax = plt.subplots(1, 5, figsize=(10, 4))
        for i in range(1, 6):
            img_resized = Functions.bilinearInterGray(
                img_gray, a=1. / 2 ** i, b=1. / 2 ** i)
            pyramid.append(img_resized)
            ax[i - 1].set_title(f"1 / {2 ** i}")
            ax[i - 1].imshow(img_resized, cmap='gray')
        plt.show()

    def question76(self):
        """顕著性マップ
        """
        img_gray = np.array(self.img_256.convert("L"))
        sal = np.zeros_like(img_gray, dtype=np.float32)

        pyramid = [img_gray.astype(np.float32)]

        for i in range(1, 6):
            img_resized = Functions.bilinearInterGray(
                img_gray, a=1. / 2 ** i, b=1. / 2 ** i)
            img_resized = Functions.bilinearInterGray(
                img_resized, a=2 ** i, b=2 ** i)
            pyramid.append(img_resized.astype(np.float32))

        pyramid_n = len(pyramid)

        for i in range(pyramid_n):
            for j in range(pyramid_n):
                if i == j:
                    continue
                sal += np.abs(pyramid[i] - pyramid[j])

        sal /= sal.max()
        sal *= 255
        sal = sal.astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("saliency")
        ax[1].imshow(sal, cmap="gray")
        plt.show()

    def question77(self):
        """ガボールフィルタ
        """
        gabor = Functions.gaborFilter(
            K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0)

        plt.imshow(gabor, cmap="gray")
        plt.show()

    def question78(self):
        """ガボールフィルタの回転
        """

        fig, ax = plt.subplots(1, 4, figsize=(10, 2))

        for i, a in enumerate([0, 45, 90, 135]):
            gabor = Functions.gaborFilter(
                K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=a)
            ax[i].set_title(f"rot = {a}")
            ax[i].imshow(gabor, cmap='gray')

        plt.show()

    def question79(self):
        """ガボールフィルタによるエッジ抽出
        """
        img_gray = np.array(self.img_256.convert("L"))

        fig, ax = plt.subplots(1, 5, figsize=(12, 4))
        ax[0].set_title("input")
        ax[0].imshow(img_gray, cmap="gray")

        for i, a in enumerate([0, 45, 90, 135]):
            img_gabor = Functions.gaborFiltering(
                img_gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=a)
            ax[i + 1].set_title(f"rot = {a}")
            ax[i + 1].imshow(img_gabor, cmap='gray')

        plt.show()

    def question80(self):
        """ガボールフィルタによる特徴抽出
        """
        img_gray = np.array(self.img_256.convert("L"))
        img_gabor = Functions.gaborFiltering(
            img_gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=0)
        out = np.zeros_like(img_gabor, dtype=np.float32)

        for i, a in enumerate([0, 45, 90, 135]):
            img_gabor = Functions.gaborFiltering(
                img_gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=a)
            out += img_gabor.astype(np.float32)

        out = out / out.max() * 255
        out = out.astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("output")
        ax[1].imshow(out, cmap="gray")
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

    # mainObject.question71()
    # mainObject.question72()
    # mainObject.question73()
    # mainObject.question74()
    # mainObject.question75()
    # mainObject.question76()
    # mainObject.question77()
    # mainObject.question78()
    # mainObject.question79()
    mainObject.question80()


if __name__ == "__main__":
    main()
