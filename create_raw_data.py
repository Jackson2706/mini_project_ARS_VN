import os

import cv2

data_dir = "./Data"
raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
i = 0
for phase_case in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, phase_case)
    if phase_case == "conveyor_video_for_train":
        for video in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video)
            video = cv2.VideoCapture(video_path)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                if i % 10:
                    i = i + 1
                    continue
                cv2.imwrite(f"{raw_data_dir}/conveyor_{i}.jpg", frame)
                i = i + 1
            video.release()

    if phase_case == "cellphone_video_for_train":
        for object in os.listdir(folder_path):
            object_path = os.path.join(folder_path, object)
            for video in os.listdir(object_path):
                video_path = os.path.join(object_path, video)
                video = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    if i % 100:
                        i = i + 1
                        continue
                    cv2.imwrite(f"{raw_data_dir}/cellphone_{i}.jpg", frame)
                    i = i + 1
                video.release()
