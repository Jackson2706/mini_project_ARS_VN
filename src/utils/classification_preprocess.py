from cv2 import COLOR_BGR2GRAY, cvtColor, resize
from numpy import argsort, array, float64, maximum, minimum, where
from skimage.feature import hog
from skimage.transform import resize

"""
    Extract images from RGB images to 1D images with using HoG method
    @param: image_path, for example: "./images/abc.jpg"
    @return: a hoG image

"""


def preprocess_img(img):
    if len(img.shape) > 2:
        img = cvtColor(img, COLOR_BGR2GRAY)
    img = img.astype(float64)
    resized_img = resize(img, output_shape=(32, 32), anti_aliasing=True)
    hog_features = hog(
        resized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        feature_vector=True,
    )
    return hog_features


def crop_object(img, bbox):
    x_min, y_min, x_max, y_max = bbox
    object_image = img[y_min:y_max, x_min:x_max]
    return object_image


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    x1 = maximum(box[0], boxes[:, 0])
    y1 = maximum(box[1], boxes[:, 1])
    x2 = minimum(box[2], boxes[:, 2])
    y2 = minimum(box[3], boxes[:, 3])

    intersection = maximum((x2 - x1), 0) * maximum((y2 - y1), 0)

    union = box_area + boxes_area[:] - intersection

    iou = intersection * 1.0 / union

    return iou


def nms(bboxes, iou_threshold):
    if not bboxes:
        return []
    scores = array([bbox[5] for bbox in bboxes])
    sorted_indices = argsort(scores)[::-1]

    xmin = array([bbox[0] for bbox in bboxes])
    ymin = array([bbox[1] for bbox in bboxes])
    xmax = array([bbox[2] for bbox in bboxes])
    ymax = array([bbox[3] for bbox in bboxes])

    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    keep = []

    while sorted_indices.size > 0:
        i = sorted_indices[0]
        keep.append(i)

        iou = compute_iou(
            [xmin[i], ymin[i], xmax[i], ymax[i]],
            array(
                [
                    xmin[sorted_indices[1:]],
                    ymin[sorted_indices[1:]],
                    xmax[sorted_indices[1:]],
                    ymax[sorted_indices[1:]],
                ]
            ).T,
            areas[i],
            areas[sorted_indices[1:]],
        )

        idx_to_keep = where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[idx_to_keep + 1]
    return [bboxes[i] for i in keep]
