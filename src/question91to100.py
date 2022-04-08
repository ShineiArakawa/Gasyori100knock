"""https://github.com/yoyoyo-yo/Gasyori100knock
"""

import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io

from functions import Functions


class Main:
    test_paths = [
            "../dataset/test/akahara_1.jpg",
            "../dataset/test/akahara_2.jpg",
            "../dataset/test/akahara_3.jpg",
            "../dataset/test/madara_1.jpg",
            "../dataset/test/madara_2.jpg",
            "../dataset/test/madara_3.jpg"
        ]
    
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

    def question81(self):
        """Hessianのコーナー検出
        """
        img_thorino_gray = np.array(self.img_thorino.convert("L"))
        img_thorino_hessian = Functions.hessianCorner(img_thorino_gray)

        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].set_title("input")
        ax[0].imshow(self.img_thorino, cmap="gray")
        ax[1].set_title("output")
        ax[1].imshow(img_thorino_hessian, cmap="gray")
        plt.show()

    def question82(self):
        """Harrisのコーナー検出 (Step.1) Sobel + Gauusian
        """
        img_thorino_gray = np.array(self.img_thorino.convert("L"))

        # get difference image
        i_x2, i_y2, i_xy = Functions.sobelFiltering(img_thorino_gray)

        # gaussian filtering
        i_x2 = Functions.gaussianFiltering(i_x2, K_size=3, sigma=3)
        i_y2 = Functions.gaussianFiltering(i_y2, K_size=3, sigma=3)
        i_xy = Functions.gaussianFiltering(i_xy, K_size=3, sigma=3)

        fig, ax = plt.subplots(1, 4, figsize=(15, 3))
        ax[0].set_title("input")
        ax[0].imshow(self.img_thorino, cmap="gray")
        ax[1].set_title("i_x2")
        ax[1].imshow(i_x2, cmap="gray")
        ax[2].set_title("i_y2")
        ax[2].imshow(i_y2, cmap="gray")
        ax[3].set_title("i_xy")
        ax[3].imshow(i_xy, cmap="gray")
        plt.show()

    def question83(self):
        """Harrisのコーナー検出 (Step.2) コーナー検出
        """
        img_thorino_gray = np.array(self.img_thorino.convert("L"))
        i_x2, i_y2, i_xy = Functions.sobelFiltering(img_thorino_gray)
        i_x2 = Functions.gaussianFiltering(i_x2, K_size=3, sigma=3)
        i_y2 = Functions.gaussianFiltering(i_y2, K_size=3, sigma=3)
        i_xy = Functions.gaussianFiltering(i_xy, K_size=3, sigma=3)

        out = Functions.cornerDetect(img_thorino_gray, i_x2, i_y2, i_xy)

        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].set_title("input")
        ax[0].imshow(self.img_thorino, cmap="gray")
        ax[1].set_title("output")
        ax[1].imshow(out, cmap="gray")
        plt.show()

    def question84(self):
        """簡単な画像認識 (Step.1) 減色化 + ヒストグラム\n
        """
        # get database
        db, train_paths = Functions.getDB()

        fig, ax = plt.subplots(1, 6, figsize=(15, 2))

        for i in range(len(db)):
            ax[i].set_title(train_paths[i].split("/")[-1])
            ax[i].bar(np.arange(12), db[i, :-1])

        plt.show()

    def question85(self):
        """簡単な画像認識 (Step.2) クラス判別
        """
        db, train_paths = Functions.getDB()

        fig, ax = plt.subplots(1, 6, figsize=(15, 2))
        for i in range(len(db)):
            ax[i].set_title(train_paths[i].split("/")[-1])
            ax[i].bar(np.arange(12), db[i, :-1])
        plt.show()

        for path in self.test_paths:
            img = io.imread(path)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = Functions.dicColor(img)
            feat = Functions.getFeature(img)

            db_diff = np.abs(db[:, :-1] - feat)
            distances = db_diff.sum(axis=1)
            nearest = distances.argmin()
            pred_cls = db[nearest, -1]
            label = "akahara" if pred_cls == 0 else "madara"

            print(path.split("/")[-1], ", pred >>", label)

    def question86(self):
        """簡単な画像認識 (Step.3) 評価(Accuracy)
        """
        raise NotImplementedError()

    def question87(self):
        """簡単な画像認識 (Step.4) k-NN
        """
        k = 3

        from collections import Counter
        db, train_paths = Functions.getDB()
        
        for path in self.test_paths:
            img = io.imread(path)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            img = Functions.dicColor(img)
            feat = Functions.getFeature(img)

            db_diff = np.abs(db[:, :-1] - feat)
            distances = db_diff.sum(axis=1)
            nearest = distances.argsort()[:k]
            pred_cls = db[nearest, -1]
            
            counts = Counter(pred_cls).most_common()
            label = "akahara" if counts[0][0] == 0 else "madara"
            
            print(path.split("/")[-1], ", pred >>", label)
            
    def question88(self):
        """K-means (Step.1) 重心作成
        """
        db, train_paths = Functions.getDB()
        Functions.assignLabelInit(db, train_paths)

    def question89(self):
        """K-means (Step.2) クラスタリング
        """
        db, train_paths = Functions.getDB()
        Functions.assignLabelInit(db, train_paths)
        Functions.kmeans(db, train_paths)

    def question90(self):
        """K-means データを増やす
        """
        db2, train_paths2 = Functions.getDBAll()
        print("\nkmeans")
        Functions.assignLabelInit(db2, train_paths2)
        Functions.kmeans(db2, train_paths2)

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

    # mainObject.question81()
    # mainObject.question82()
    # mainObject.question83()
    # mainObject.question84()
    # mainObject.question85()
    # mainObject.question86()
    # mainObject.question87()
    # mainObject.question88()
    # mainObject.question89()
    mainObject.question90()


if __name__ == "__main__":
    main()
