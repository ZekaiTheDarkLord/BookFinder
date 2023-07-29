import cv2
import super_gradients
import os


def get_bounding_box_for_single_book(filePath):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()

    newBounds = yolo_nas.predict(image).show()

    booksBounds = []

    for bounds in newBounds:
        if bounds[2] - bounds[0] < bounds[3] - bounds[1]:
            booksBounds.append(bounds)

    print(booksBounds)
    return booksBounds


def get_bounding_box_for_bookshell(filePath):
    # images = []
    # filePath = "PaddleSeg/pictures/4.jpg"

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()

    newBounds = yolo_nas.predict(image).show()

    booksBounds = []

    for bounds in newBounds:
        if bounds[2] - bounds[0] > bounds[3] - bounds[1]:
            booksBounds.append(bounds)

    print(booksBounds)
    return booksBounds
