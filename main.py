from time import time

import cv2
import numpy as np
import torch

from src.model.CNN import CNN
from src.model.MLP import MLP

# __________________________________________
# test video
test_vid = "./test.avi"
# the threshold take into consideration
# count from top edge
frame_thres = 7

# lower boundary is at the top ... of the frame
frame_ratio = 1 / 2

# Choose model
model = CNN()
# __________________________________________

model.load_state_dict(torch.load("weights/best_cnn.pt", map_location="cpu"))

fgbg = cv2.createBackgroundSubtractorMOG2()
capture = cv2.VideoCapture(test_vid)

# where we draw the guide line
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
lower_boundary = int(height * frame_ratio)

cur_center = 0
frame_count = 0
obj_in_frame = False
this_obj = False
time_proc = 0
# tensor save sum of all prediction
obj_classes = np.zeros(10)
prelabel = 10
label = 10


# label --  count of passed object
decode_label = {
    0: ["Dilmah", 0],
    1: ["G7 caffe", 0],
    2: ["Jack Jill", 0],
    3: ["Karo", 0],
    4: ["Nestea atiso", 0],
    5: ["Nestea chanh", 0],
    6: ["Nestea hoa qua", 0],
    7: ["Orion", 0],
    8: ["Tipo", 0],
    9: ["Y40", 0],
    10: ["NONE"],
}


# return a tensor
def classification(img):
    # pre CNN
    image = cv2.resize(img, (128, 128))
    image = torch.from_numpy(image[:, :, (2, 1, 0)]).permute(2, 0, 1)
    image = image.to(torch.float32)

    # pre MLP
    # image = preprocess_img(img)
    # image = torch.FloatTensor(image)

    # classification
    image = image.unsqueeze(0)
    model_result = model(image)
    # result_class = torch.argmax(model_result)
    return model_result


def putText(img, text, n_line, header=False):
    color = (150, 0, 0)
    if header:
        color = (0, 0, 255)
        text = " " * 10 + text
    cv2.putText(
        img, text, (100, 50 * n_line), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
    )


while True:
    contours_now = []
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
        if cv2.contourArea(c) > 50000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cur_center = y + h // 2

            # when object reach guide line until the threshold
            if (cur_center < lower_boundary) and (frame_count < frame_thres):
                obj_in_frame = True
                obj = obj[y : y + h, x : x + w]
                frame_count += 1

                start = time()

                label_result = int(torch.argmax(classification(obj)))
                obj_classes[label_result] += 1
                label = np.argmax(obj_classes)

                end = time()
                time_proc = round((end - start), 5)

        # when object go out of the frame
        elif obj_in_frame:
            # get the final label
            f_label = label
            decode_label[f_label][1] += 1

            # reset parameters
            obj_in_frame = False
            prelabel = f_label
            label = 10
            obj_classes = np.zeros(10)
            frame_count = 0

    # draw guide line
    cv2.line(
        frame,
        (0, lower_boundary),
        (frame.shape[1], lower_boundary),
        (0, 255, 255),
        2,
    )

    # Interface:
    cv2.rectangle(frame, (50, 50), (850, 250), (238, 215, 189), -1)
    cv2.rectangle(frame, (50, 300), (850, 900), (238, 215, 189), -1)
    putText(frame, "Detect", 2, True)
    putText(frame, "Counter", 7, True)

    # show number of obj:
    putText(frame, "Prev obj: " + str(decode_label[prelabel][0]), 3)
    putText(frame, "Current obj: " + str(decode_label[label][0]), 4)
    for i in range(10):
        putText(
            frame, decode_label[i][0] + ": " + str(decode_label[i][1]), i + 8
        )

    putText(frame, " " * 30 + "Time processing: " + str(time_proc), 2, True)
    # show the current frame and the fg masks
    frame = cv2.resize(frame, (680, 680))
    cv2.imshow("Frame", frame)

    # press 'q' to stop
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
