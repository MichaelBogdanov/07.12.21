import os
import face_recognition
import cv2
import numpy as np
import datetime as dt
import easygui
import requests


# Выбираем видео
video_capture = cv2.VideoCapture(easygui.fileopenbox())
# Инициализируем файл с выходными данными
file = open("discovered.txt", "a+");
# Создаём переменную времени и даты
now = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
# Создаём список обнаруженных имён
names = []
# Создаём списки для знакомых лиц
known_face = []
known_face_encodings = []
known_face_names = []
for i in os.listdir("faces"):
    image = face_recognition.load_image_file("faces/"+i)
    known_face.append(image)
    known_face_names.append(i.split(".")[0])
    known_face_encodings.append(face_recognition.face_encodings(image)[0])


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Считываем кадры видео
    ret, frame = video_capture.read()

    # Заканчиваем, если обработаны все кадры
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        name = "Unknown"
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"


            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Записываем обнаруженного в файл и отправляем данные на сервер
    session = requests.Session()
    if len(names) > 0:
        if name not in names and name != "Unknown":
            file.write(f"{now} - {name}\n")
            names.append(name)
            session.get(f"http://lordprs0.beget.tech/cam_detect.php?name={name}")
    else:
        if name != "Unknown":
            file.write(f"{now} - {name}\n")
            names.append(name)
            session.get(f"http://lordprs0.beget.tech/cam_detect.php?name={name}")


    # Завершаем работу при нажатии «q»
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Всё готово, закрываем окна и потоки
file.close()
video_capture.release()
cv2.destroyAllWindows()