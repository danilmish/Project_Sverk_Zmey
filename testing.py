import cv2
import numpy as np

def nothing(x):
  pass

# Укажите путь к вашей картинке
image_path = 'testi.png'
frame = cv2.imread(image_path)

if frame is None:
  print(f'Ошибка: не удалось загрузить изображение "{image_path}".')
  exit()

# Создаем окно с ползунками
cv2.namedWindow('Tracking')
cv2.createTrackbar('LH', 'Tracking', 35, 180, nothing)
cv2.createTrackbar('LS', 'Tracking', 20, 255, nothing)
cv2.createTrackbar('LV', 'Tracking', 40, 255, nothing)
cv2.createTrackbar('UH', 'Tracking', 85, 180, nothing)
cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

# Переводим исходную картинку в HSV один раз для скорости
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

print('Крутите ползунки в окне "Tracking". Для выхода нажмите клавишу "q".')

while True:
  # Считываем текущие позиции ползунков
  lh = cv2.getTrackbarPos('LH', 'Tracking')
  ls = cv2.getTrackbarPos('LS', 'Tracking')
  lv = cv2.getTrackbarPos('LV', 'Tracking')
  uh = cv2.getTrackbarPos('UH', 'Tracking')
  us = cv2.getTrackbarPos('US', 'Tracking')
  uv = cv2.getTrackbarPos('UV', 'Tracking')

  lower_g = np.array([lh, ls, lv])
  upper_g = np.array([uh, us, uv])

  # Создаем маску и результат
  mask = cv2.inRange(hsv, lower_g, upper_g)
  result = cv2.bitwise_and(frame, frame, mask=mask)

  # Показываем окна
  cv2.imshow('Original', frame)
  cv2.imshow('Mask', mask)
  cv2.imshow('Result', result)

  # Выход по нажатию клавиши 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    print(f'Итоговые значения порогов:')
    print(f'lower_green = np.array([{lh}, {ls}, {lv}])')
    print(f'upper_green = np.array([{uh}, {us}, {uv}])')
    break

cv2.destroyAllWindows()