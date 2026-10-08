import cv2
import numpy as np

image = cv2.imread("dusuk_kontrast.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Hata: Resim bulunamadi!")
    exit()

h, w = image.shape

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
opencv_sonuc = clahe.apply(image)
cv2.imwrite("clahe_opencv.jpg", opencv_sonuc)

manuel_sonuc = np.zeros((h, w), dtype=np.uint8)

grid_y, grid_x = 8, 8
tile_h = h // grid_y
tile_w = w // grid_x
clip_limit = 40

for gy in range(grid_y):
    for gx in range(grid_x):
        y1 = gy * tile_h
        y2 = (gy + 1) * tile_h if gy < grid_y - 1 else h
        x1 = gx * tile_w
        x2 = (gx + 1) * tile_w if gx < grid_x - 1 else w
        
        blok = image[y1:y2, x1:x2]
        
        hist = [0] * 256
        for piksel in blok.ravel():
            hist[piksel] += 1
            
        artan = 0
        for i in range(256):
            if hist[i] > clip_limit:
                artan += (hist[i] - clip_limit)
                hist[i] = clip_limit
        
        ekstra = artan // 256
        for i in range(256):
            hist[i] += ekstra

        cdf = [0] * 256
        toplam = 0
        for i in range(256):
            toplam += hist[i]
            cdf[i] = toplam
            
        toplam_piksel = blok.size
        lut = [0] * 256
        for v in range(256):
            lut[v] = int(round((cdf[v] / float(toplam_piksel)) * 255))
            
        for y in range(y2 - y1):
            for x in range(x2 - x1):
                manuel_sonuc[y1 + y, x1 + x] = lut[blok[y, x]]

cv2.imwrite("clahe_manuel.jpg", manuel_sonuc)
print("Islem bitti! clahe_opencv.jpg ve clahe_manuel.jpg kaydedildi.")

cv2.imshow("1. Orijinal", image)
cv2.imshow("2. OpenCV CLAHE", opencv_sonuc)
cv2.imshow("3. Manuel CLAHE", manuel_sonuc)

cv2.waitKey(0)
cv2.destroyAllWindows()
