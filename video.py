import cv2
import numpy as np

# Открываем видеофайл
video_path = 'rec.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f'Ошибка: не удалось открыть видеофайл {video_path}')
    exit()

# Пороги для зеленого цвета в пространстве HSV
lower_green = np.array([30, 40, 20])
upper_green = np.array([85, 255, 255])

lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([35, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])


lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])






gamma = 1.5
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Видео закончилось или возникла ошибка чтения.')
        break

    # 1. Сначала осветляем исходный кадр (в BGR)
    bright_img = cv2.LUT(frame, table)

    # 2. Переводим ИМЕННО ОСВЕТЛЕННЫЙ кадр в HSV
    hsv = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HSV)

    # 3. Применяем inRange к HSV-изображению
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    final_mask = cv2.bitwise_or(mask_yellow, mask_green)
    final_mask = cv2.bitwise_or(final_mask, mask_red)








    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ищем контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            # Рисуем контуры и прямоугольник НА ОСВЕТЛЕННОМ кадре (bright_img), 
            # чтобы сразу видеть результат работы
            cv2.drawContours(bright_img, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(bright_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Выводим результирующий поток
    cv2.imshow('Green Color Tracking', bright_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()