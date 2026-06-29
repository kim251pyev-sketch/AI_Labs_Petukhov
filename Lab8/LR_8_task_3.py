import cv2
import numpy as np

img = cv2.imread("Petukhov.jpg")

print(f"Розмір зображення: {img.shape}")

y1, y2 = 50, 450
x1, x2 = 50, 350
imgCropped = img[y1:y2, x1:x2]

cv2.imshow("Original Image", img)
cv2.imshow("Cropped Face", imgCropped)

cv2.waitKey(0)
cv2.destroyAllWindows()