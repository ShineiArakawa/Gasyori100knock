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
from nn import NeuralNet, NeuralNet2, sliding_window_classify


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
        self.img_madara = Image.open(
            "../dataset/images/madara_256x256.png"
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
        self.imgArray_madara = np.array(self.img_madara,
                                      dtype=np.float64)
        # =================================================================================
        pass

    def question91(self):
        """K-meansによる減色処理 (Step.1) 色の距離によるクラス分類
        """
        out = Functions.colorKmeansStep1(self.imgArray_512)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_512)
        ax[1].set_title("output")
        ax[1].imshow(out)
        plt.show()

    def question92(self):
        """K-meansによる減色処理 (Step.2) 減色処理
        """
        out = Functions.colorKmeans(self.imgArray_256)
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("output")
        ax[1].imshow(out)
        plt.show()
        plt.clf()
        
        out = Functions.colorKmeans(self.imgArray_madara)
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_madara)
        ax[1].set_title("output")
        ax[1].imshow(out)
        plt.show()

    def question93(self):
        """機械学習の学習データの用意 (Step.1) IoUの計算
        """
        # [x1, y1, x2, y2]
        a = np.array((50, 50, 150, 150), dtype=np.float32)
        b = np.array((60, 60, 170, 160), dtype=np.float32)

        print(Functions.getIOU(a, b))

    def question94(self):
        """機械学習の学習データの用意 (Step.2) ランダムクラッピング
        """
        # gt bounding box
        gt = np.array((130, 120, 190, 180), dtype=np.float32)

        crops, labels = Functions.crop_bbox(self.imgArray_256, gt)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("output")
        ax[1].imshow(self.img_256)

        for i in range(len(crops)):
            c = "red" if labels[i] == 1 else "blue"
            x1, y1, x2, y2 = crops[i]
            ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=0.5))

        ax[1].add_patch(plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], fill=False, edgecolor="green", linewidth=2))
            
        plt.show()

    def question95(self):
        """ニューラルネットワーク (Step.1) 使ってみる
        """
        train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
        train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

        nn = NeuralNet()

        # train
        for i in range(1000):
            nn.forward(train_x)
            nn.train(train_x, train_t)

        # test
        for j in range(4):
            x = train_x[j]
            t = train_t[j]
            print("in:", x, "pred:", nn.forward(x))
            
        nn = NeuralNet2()

        # train
        for i in range(1000):
            nn.forward(train_x)
            nn.train(train_x, train_t)

        # test
        for j in range(4):
            x = train_x[j]
            t = train_t[j]
            print("in:", x, "pred:", nn.forward(x))

    def question96(self):
        """ニューラルネットワーク (Step.2) 学習
        """
        # prepare gt bounding box
        gt = np.array((130, 120, 190, 180), dtype=np.float32)

        # get database
        db = Functions.make_dataset(self.imgArray_256, gt)

        # train neural network
        # get input feature dimension
        input_dim = db.shape[1] - 1
        train_x = db[:, :input_dim]
        train_t = db[:, -1][..., None]

        nn = NeuralNet(in_dim=input_dim, lr=0.01)

        # training
        for i in range(10_000):
            nn.forward(train_x)
            nn.train(train_x, train_t)

        # test
        accuracy_n = 0.

        for x, t in zip(train_x, train_t):
            prob = nn.forward(x)

            # count accuracy
            pred = 1 if prob >= 0.5 else 0
            if t == pred:
                accuracy_n += 1

        accuracy = accuracy_n / len(db)
        print("Accuracy >> {} ({} / {})".format(accuracy, accuracy_n, len(db)))

    def question97(self):
        """簡単な物体検出 (Step.1) スライディングウィンドウ + HOG
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
            
    def question98(self):
        """簡単な物体検出 (Step.2) スライディングウィンドウ + NN
        """
        gt = np.array((130, 120, 190, 180), dtype=np.float32)
        
        detects = sliding_window_classify(self.imgArray_256)

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("output")
        ax[1].imshow(self.img_256)

        for i in range(len(detects)):
            x1, y1, x2, y2, score = detects[i]
            ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=0.5))

        ax[1].add_patch(plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], fill=False, edgecolor="green", linewidth=2))
            
        plt.show()

    def question99(self):
        """簡単な物体検出 (Step.3) Non-Maximum Suppression
        """
        gt = np.array((130, 120, 190, 180), dtype=np.float32)
        detects = sliding_window_classify(self.imgArray_256)
        selected_inds = Functions.nms(detects, iou_th=0.25)
        _detects = detects[selected_inds]

        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        ax[0].set_title("input")
        ax[0].imshow(self.img_256)
        ax[1].set_title("output")
        ax[1].imshow(self.img_256)

        for d in _detects:
            x1, y1, x2, y2, score = d
            ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=0.5))

        ax[1].add_patch(plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], fill=False, edgecolor="green", linewidth=2))
            
        plt.show()

    def question100(self):
        """簡単な物体検出 (Step.4) 評価 Precision, Recall, F-score, mAP
        """
        detects = sliding_window_classify(self.imgArray_256)
        selected_inds = Functions.nms(detects, iou_th=0.25)
        _detects = detects[selected_inds]

        gt = np.array([[130, 120, 190, 180], ], dtype=np.float32)

        # Recall, Precision, F-score
        iou_th = 0.5

        Rs = np.zeros((len(gt)))
        Ps = np.zeros((len(_detects)))

        for i, g in enumerate(gt):
            iou_x1 = np.maximum(g[0], _detects[:, 0])
            iou_y1 = np.maximum(g[1], _detects[:, 1])
            iou_x2 = np.minimum(g[2], _detects[:, 2])
            iou_y2 = np.minimum(g[3], _detects[:, 3])
            iou_s = np.maximum(0, iou_x2 - iou_x1) * np.maximum(0, iou_y2 - iou_y1)
            g_s = (g[2] - g[0]) * (g[3] - g[1])
            d_s = (_detects[:, 2] - _detects[:, 0]) * (_detects[:, 3] - _detects[:, 1])
            ious = iou_s / (g_s + d_s - iou_s)
            
            Rs[i] = 1 if np.sum(ious >= iou_th) > 0 else 0
            Ps[ious >= iou_th] = 1

        R = np.mean(Rs)
        P = np.mean(Ps)
        F = (2 * P * R) / (P + R) 

        print("Recall >> {:.2f} ({} / {})".format(R, np.sum(Rs), len(Rs)))
        print("Precision >> {:.2f} ({} / {})".format(P, np.sum(Ps), len(Ps)))
        print("F-score >> ", F)

        # mAP
        mAP = np.mean([np.sum(Ps[:i+1]) / (i + 1) for i in range(len(_detects)) if Ps[i] == 1])
        print("mAP >>", mAP)

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

    # mainObject.question91()
    # mainObject.question92()
    # mainObject.question93()
    # mainObject.question94()
    # mainObject.question95()
    # mainObject.question96()
    # mainObject.question97()
    # mainObject.question98()
    # mainObject.question99()
    mainObject.question100()


if __name__ == "__main__":
    main()
