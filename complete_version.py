import cv2
import sverk_interfaces
import time
import cv2
import numpy as np

drone = sverk_interfaces.init(Nodename="vision_stream")




lower_green = np.array([30, 40, 20])
upper_green = np.array([85, 255, 255])

lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([35, 255, 255])

# lower_red1 = np.array([0, 5, 3])
# upper_red1 = np.array([10, 255, 255])


# lower_red2 = np.array([170, 5, 3])
# upper_red2 = np.array([180, 255, 255])

gamma = 1.5
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')


MIN_DETECTION_TIME = 1.0  # Минимальное время удержания контура (в секундах)
POSITION_THRESHOLD = 50   # Максимальный сдвиг центра в пикселях между кадрами


tracked_candidates = []

def proccess_frame(frame):
    global tracked_candidates

    current_time = time.time()

    frame = cv2.resize(frame, (640, 360))

    # 1. Осветляем кадр и переводим в HSV
    bright_img = cv2.LUT(frame, table)
    hsv = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HSV)

    # 2. Маскирование
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    
    # mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    final_mask = cv2.bitwise_or(mask_yellow, mask_green)
    # final_mask = cv2.bitwise_or(final_mask, mask_red)


    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_frame_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 10000:  # Геометрические фильтры
            continue

        rect = cv2.minAreaRect(cnt)
        (center, (width, height), angle) = rect

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.5:  # Фильтр длинных линий
            continue

        box_area = width * height
        extent = area / box_area if box_area > 0 else 0
        if extent < 0.3:
            continue

        cx, cy = int(center[0]), int(center[1])
        current_frame_candidates.append({'cx': cx, 'cy': cy, 'cnt': cnt})

    # 4. Временной фильтр (трекинг кандидатов)
    updated_tracked = []

    for item in current_frame_candidates:
        cx, cy, cnt = item['cx'], item['cy'], item['cnt']
        matched = False

        # Проверяем, совпадает ли объект с кем-то из прошлых кадров
        for cand in tracked_candidates:
            prev_cx, prev_cy = cand['center']
            dist = np.hypot(cx - prev_cx, cy - prev_cy)

            if dist < POSITION_THRESHOLD:
                matched = True
                cand['center'] = (cx, cy)
                cand['last_seen'] = current_time

                duration = current_time - cand['first_seen']

                # Проверка порога 1 секунды
                if duration >= MIN_DETECTION_TIME:
                    # Если яблоко только что подтвердилось — пишем в терминал
                    if not cand['confirmed']:
                        cand['confirmed'] = True
                        print(f"[УСПЕХ] ЯБЛОКО НАЙДЕНО! (Контур удержан {duration:.1f} сек в точке X:{cx}, Y:{cy})")

                    # Подтвержденное яблоко обводим ЗЕЛЕНЫМ
                    cv2.drawContours(bright_img, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(bright_img, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(bright_img, f"Apple ({duration:.1f}s)", (cx - 30, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Кандидат еще набирает время — обводим ЖЕЛТЫМ
                    cv2.drawContours(bright_img, [cnt], -1, (0, 255, 255), 1)
                    cv2.putText(bright_img, f"Tracking: {duration:.1f}s", (cx - 30, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                updated_tracked.append(cand)
                break

        # Если это совершенно новый контур (первый раз в кадре)
        if not matched:
            updated_tracked.append({
                'center': (cx, cy),
                'first_seen': current_time,
                'last_seen': current_time,
                'confirmed': False
            })

    # Удаляем из памяти кандидатов, которые пропали из кадра дольше чем на 0.3 секунды
    tracked_candidates = [
        c for c in updated_tracked 
        if (current_time - c['last_seen']) < 0.3
    ]

    drone.image.publish(bright_img)




drone.image.stream(proccess_frame)