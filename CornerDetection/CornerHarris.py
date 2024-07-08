import cv2
import numpy as np

# Đọc ảnh và chuyển sang grayscale
img = cv2.imread('sampleHarris.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Áp dụng Harris Corner Detector
block_size = 2
ksize = 3
k = 0.04
dst = cv2.cornerHarris(gray, block_size, ksize, k)

# Kết quả được đánh dấu trên ảnh gốc
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Hiển thị kết quả
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
