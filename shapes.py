import cv2


def get_object_type(corner_len, width, height):
    if corner_len == 3:
        object_type = "Tri"
    elif corner_len == 4:
        asp_ratio = width / float(height)
        if 0.98 < asp_ratio < 1.03:
            object_type = "Square"
        else:
            object_type = "Rectangle"
    elif corner_len > 4:
        object_type = "Circles"
    else:
        object_type = "None"
    return object_type


def detect_objects(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # avoid noises
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            # determine length of contour
            contour_length = cv2.arcLength(cnt, True)
            # approximate the corner points
            approx_corner = cv2.approxPolyDP(cnt, 0.02 * contour_length, True)
            # create bounding box around the object
            x, y, w, h = cv2.boundingRect(approx_corner)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # determine object type
            obj_type = get_object_type(len(approx_corner), w, h)
            cv2.putText(imgContour, obj_type, (x + 10, y + h - 10), cv2.QT_FONT_NORMAL, 0.7, (0, 0, 0), 2)


path = 'assets/shapes.png'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)

detect_objects(imgCanny)

cv2.imshow("Original", img)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Contour", imgContour)

cv2.waitKey(0)
