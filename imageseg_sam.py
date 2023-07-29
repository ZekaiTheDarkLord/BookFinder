import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


def show_flip_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])
    h, w = mask.shape[-2:]
    mask_image = ~mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_mask(file_path, input_box):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sys.path.append("..")

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    masks = []

    for box in input_box:
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        masks.append(mask)

    return image, masks


def get_selected(image, mask):
    color = np.array([1, 1, 1])
    h, w = mask.shape[-2:]

    reshape_mask = mask.reshape(h, w, 1)
    reshape_color = color.reshape(1, 1, -1)

    mask_image = np.uint8(reshape_mask * reshape_color)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    res = mask_image * image

    return res


file_path = "PaddleSeg/pictures/4.jpg"
input_box = np.array([[655, 1452, 2411, 2075], [578, 711, 2437, 1370], [698, 2176, 2368, 2689],
                      [507, 3, 2555, 539], [703, 2848, 2365, 3262]])

image, masks = get_mask(file_path, input_box)

res = get_selected(image, masks[0])

cv2.imwrite('output.jpg', res)
