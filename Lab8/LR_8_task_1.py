import cv2

img = cv2.imread("Petukhov.jpg")

if img is not None:
    cv2.imshow("Petukhov", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Помилка: не вдалося знайти або відкрити файл 'Petukhov.jpg'.")
    print("Перевірте, чи файл знаходиться в тій же директорії, що й скрипт.")