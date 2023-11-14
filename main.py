import cv2

fgbg = cv2.createBackgroundSubtractorMOG2()
capture = cv2.VideoCapture("test.avi")

line = 650

contours_previous = []
people_out = 0
people_in = 0
contours_now = []

while True:
    contours_now = []
    (grabbed, frame) = capture.read()

    if not grabbed:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = fgbg.apply(gray)

    fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

    fgMask = cv2.dilate(fgMask, None, iterations=2)
    fgMask = cv2.erode(fgMask, None, iterations=2)

    contours_list, hierarchy = cv2.findContours(
        fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours
    sorted_contours = sorted(contours_list, key=cv2.contourArea, reverse=True)
    for c in contours_list:
        if cv2.contourArea(c) < 50000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.line(frame, (0, line), (frame.shape[1], line), (0, 255, 255), 2)

    # show the current frame and the fg masks
    frame = cv2.resize(frame, (680, 680))
    fgMask = cv2.resize(fgMask, (680, 680))
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)
    ## [show]

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
