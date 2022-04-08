

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

    def question31(self):
        """スキュー
        """
        h, w, c = self.imgArray_512.shape
        img_affine_1 = Functions.affine(self.imgArray_512,
                                        [[1, 30 / h, 0],
                                         [0, 1, 0]],
                                        (h, w*1.2))
        img_affine_2 = Functions.affine(self.imgArray_512,
                                        [[1, 0, 0],
                                         [30 / w, 1, 0]],
                                        (h * 1.2, w))
        img_affine_3 = Functions.affine(self.imgArray_512,
                                        [[1, 30 / h, 0],
                                         [30 / w, 1, 0]],
                                        (h * 1.2, w*1.2))

        self.__plotImages([self.imgArray_512,
                           img_affine_1,
                           img_affine_2,
                           img_affine_3])

    def question32(self):
        """離散フーリエ変換
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_128)
        freq = Functions.DFT(grayScale)
        powerSpectrum = (np.abs(freq) / np.abs(freq).max() * 255).astype(np.uint8)

        print(powerSpectrum)
        img_IDFT = Functions.IDFT(freq)
        
        self.__plotImages([grayScale, powerSpectrum, img_IDFT])

    def question33(self):
        """ローパスフィルター
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_128)
        freq = Functions.DFT(grayScale)
        
        filterd = Functions.lowPassFilter(freq)
        img_IDFT = Functions.IDFT(filterd)
        
        self.__plotImages([grayScale, img_IDFT])

    def question34(self):
        """ハイパスフィルター
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_128)
        freq = Functions.DFT(grayScale)
        
        filterd = Functions.highPassFilter(freq)
        img_IDFT = Functions.IDFT(filterd)
        
        self.__plotImages([grayScale, img_IDFT])

    def question35(self):
        """バンドパスフィルター
        """
        grayScale = Functions.grayScaleTransform(self.imgArray_128)
        freq = Functions.DFT(grayScale)
        
        filterd = Functions.bandPassFilter(freq)
        img_IDFT = Functions.IDFT(filterd)
        
        self.__plotImages([grayScale, img_IDFT])

    def question36(self):
        """離散コサイン変換
        """
        freq = Functions.DCT(self.imgArray_128)
        img_IDCT = Functions.IDCT(freq)
        
        self.__plotImages([self.imgArray_128, freq, img_IDCT])

    def question37(self):
        """PSNR
        """
        freq = Functions.DCT(self.imgArray_128, T=8)
        img_IDCT_8 = Functions.IDCT(freq, T=8, K=8)
        
        print("T= 8, N= 8 ===============================================")
        print("MSE =", Functions.mse(self.imgArray_128, img_IDCT_8))
        print("PSNR =", Functions.PSNR(self.imgArray_128, img_IDCT_8))
        print("Bitrate =", Functions.bitrate(8, 8))
        
        freq = Functions.DCT(self.imgArray_128, T=8)
        img_IDCT_4 = Functions.IDCT(freq, T=8, K=4)
        
        print("T= 8, N= 4 ===============================================")
        print("MSE =", Functions.mse(self.imgArray_128, img_IDCT_4))
        print("PSNR =", Functions.PSNR(self.imgArray_128, img_IDCT_4))
        print("Bitrate =", Functions.bitrate(8, 4))

        self.__plotImages([self.imgArray_128, img_IDCT_4, img_IDCT_8])
        
        
        
    def question38(self):
        """量子化
        """
        freq = Functions.DCT(self.imgArray_128, T=8)
        freq = Functions.quantization(freq)
        img_IDCT_8 = Functions.IDCT(freq, T=8, K=8)
        
        print("T= 8, N= 8 ===============================================")
        print("MSE =", Functions.mse(self.imgArray_128, img_IDCT_8))
        print("PSNR =", Functions.PSNR(self.imgArray_128, img_IDCT_8))
        print("Bitrate =", Functions.bitrate(8, 8))
        
        freq = Functions.DCT(self.imgArray_128, T=8)
        freq = Functions.quantization(freq)
        img_IDCT_4 = Functions.IDCT(freq, T=8, K=4)
        
        print("T= 8, N= 4 ===============================================")
        print("MSE =", Functions.mse(self.imgArray_128, img_IDCT_4))
        print("PSNR =", Functions.PSNR(self.imgArray_128, img_IDCT_4))
        print("Bitrate =", Functions.bitrate(8, 4))

        self.__plotImages([self.imgArray_128, img_IDCT_4, img_IDCT_8])

    def question39(self):
        """YCbCr表色系
        """
        ycbcr = Functions.rgb2ycbcr(self.imgArray_512)
        ycbcr[..., 0] *= 0.7
        out = Functions.ycbcr2rgb(ycbcr)
        self.__plotImages([self.imgArray_512, out])

    def question40(self):
        """YCbCr+DCT+量子化
        """
        toEnableCupy = True
        startTime = time.perf_counter()
        x = Functions.rgb2ycbcr(self.imgArray_512)
        x = Functions.DCT(x, toEnableCupy=toEnableCupy)
        x = Functions.quantization(x)
        x = Functions.IDCT(x, K=4, toEnableCupy=toEnableCupy)
        out = Functions.ycbcr2rgb(x)
        elapsedTime = time.perf_counter() - startTime
        print(f"elapsedTime= {elapsedTime}")
            
        self.__plotImages([self.imgArray_512, out])
        
        img = Image.fromarray(out)
        img.save("compressed.png")
        

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

    # mainObject.question31()
    # mainObject.question32()
    # mainObject.question33()
    # mainObject.question34()
    # mainObject.question35()
    # mainObject.question36()
    # mainObject.question37()
    # mainObject.question38()
    # mainObject.question39()
    mainObject.question40()


if __name__ == "__main__":
    main()
