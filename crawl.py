import os
import random
import shutil
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def delete_random_images(folder_path, max_images=500000):
    # Lấy danh sách tất cả các ảnh trong thư mục
    all_images = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    # Kiểm tra số lượng ảnh
    if len(all_images) <= max_images:
        print("Số lượng ảnh không vượt quá", max_images)
        return

    # Số lượng ảnh cần xóa
    images_to_delete = len(all_images) - max_images

    # Lấy danh sách các ảnh cần xóa ngẫu nhiên
    images_to_delete_list = random.sample(all_images, images_to_delete)

    # Xóa các ảnh cần xóa
    for image_name in images_to_delete_list:
        image_path = os.path.join(folder_path, image_name)
        os.remove(image_path)
        print(f"Đã xóa: {image_name}")


def create_folder_structure(
    input_dir,
    output_dir,
    train_size=0.8,
    valid_size=0.1,
    test_size=0.1,
    seed=42,
):
    # Tạo thư mục đầu ra nếu nó không tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách các nhãn từ thư mục gốc
    labels = os.listdir(input_dir)

    # Tạo thư mục đầu ra cho mỗi loại tập dữ liệu (train, valid, test)
    for data_type in ["train", "valid", "test"]:
        data_type_dir = os.path.join(output_dir, data_type)
        os.makedirs(data_type_dir, exist_ok=True)

        # Tạo thư mục con cho mỗi nhãn trong thư mục của từng loại tập dữ liệu
        for label in labels:
            label_dir = os.path.join(data_type_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Lấy danh sách các file hình ảnh
            image_files = os.listdir(os.path.join(input_dir, label))

            # Chia dữ liệu thành tập huấn luyện, tập kiểm tra và tập xác thực
            train, test = train_test_split(
                image_files, test_size=1 - train_size, random_state=seed
            )
            valid, test = train_test_split(
                test,
                test_size=test_size / (test_size + valid_size),
                random_state=seed,
            )

            # Copy ảnh vào các thư mục tương ứng trong thư mục đầu ra
            if data_type == "train":
                destination_folder = os.path.join(data_type_dir, label)
            elif data_type == "valid":
                destination_folder = os.path.join(data_type_dir, label)
            else:  # data_type == 'test'
                destination_folder = os.path.join(data_type_dir, label)

            for filename in eval(data_type):
                src = os.path.join(input_dir, label, filename)
                dest = os.path.join(destination_folder, filename)
                shutil.copy(src, dest)


fgbg = cv2.createBackgroundSubtractorMOG2()
data_dir = "./conveyor_video_for_train"
data = "./data"
if not os.path.exists(data):
    os.mkdir(data)

for video_name in os.listdir(data_dir):
    label = video_name.split(".")[0]
    label_folder = os.path.join(data, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    video_path = os.path.join(data_dir, video_name)
    print(video_path)
    capture = cv2.VideoCapture(video_path)
    while True:
        (grabbed, frame) = capture.read()
        obj = np.copy(frame)
        if not grabbed:
            break

        # Find contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgMask = fgbg.apply(gray)

        fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

        fgMask = cv2.dilate(fgMask, None, iterations=2)
        fgMask = cv2.erode(fgMask, None, iterations=2)

        contours_list, hierarchy = cv2.findContours(
            fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours_list) != 0:
            c = max(contours_list, key=cv2.contourArea)

            # if an object is in the frame
            if cv2.contourArea(c) > 100000:
                (x, y, w, h) = cv2.boundingRect(c)
                obj = obj[y : y + h, x : x + w]
                if obj.shape[0] < 500 or obj.shape[1] < 500:
                    continue
                cv2.imwrite(
                    f"{label_folder}/{time.time()}_{obj.shape}.jpg", obj
                )
        frame = cv2.resize(frame, (680, 680))
        cv2.imshow("Frame", frame)

        # press 'q' to stop
        if cv2.waitKey(1) == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


for class_name in os.listdir(data):
    class_path = os.path.join(data, class_name)
    num_image = len([image for image in os.listdir(class_path)])
    print(f"{class_name}: {num_image}")
    # delete_random_images(folder_path=class_path, max_images=500)

create_folder_structure(input_dir=data, output_dir="./dataset")

dataset_dir = "./dataset"
for phase in os.listdir(dataset_dir):
    phase_folder = os.path.join(dataset_dir, phase)
    for label in os.listdir(phase_folder):
        label_folder = os.path.join(phase_folder, label)
        num_image = len([img for img in os.listdir(label_folder)])
        print(f"{phase}-{label}: {num_image} samples")
