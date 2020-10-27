from colorsys import hsv_to_rgb
from random import shuffle
from skimage.color import gray2rgb
from skimage.draw import polygon
from skimage.io import imread
from skimage.io import imsave
import json
import numpy as np
import os


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colros, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: hsv_to_rgb(*c), hsv))
    shuffle(colors)
    return colors


def load_annotations(dataset_dir, subset):
    """
    """
    assert subset in ["train", "test", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)

    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = [a for a in annotations if a['regions']]

    data = []
    for a in annotations:
        img = {}
        if type(a['regions']) is dict:
            img['polygons'] = [r['shape_attributes'] for r in a['regions'].values()]
            img['label'] = [r['region_attributes']['damage_type'] for r in a ['regions'].values()]
        else:
            img['polygons'] = [r['shape_attributes'] for r in a['regions']]
            img['label'] = [r['region_attributes']['damage_type'] for r in a ['regions']]

        img_path = os.path.join(dataset_dir, a['filename'])
        height, width = (imread(img_path)).shape[:2]

        img['filename'] = a['filename']
        img['height'] = height
        img['width'] = width
        img["subset"] = subset
        data.append(img)

    return data


def load_image(image_filename, subset="train"):
    path = os.path.join(
        "../data", subset, image_filename
    )
    image = imread(path)

    # If grayscale. Convert to RGB for consistency
    if image.ndim != 3:
        image = gray2rgb(image)

    # Remove alpha channel for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    print(image.shape)
    return image


def load_mask(image_data):
    """
    Returns mask and class_IDs for one image. 
    """
    mask = np.zeros([image_data["height"], image_data["width"], len(image_data["polygons"])], dtype=np.uint8)

    for i, p in enumerate(image_data["polygons"]):
        rr, cc = polygon(p["all_points_y"], p["all_points_x"])
        mask[rr, cc, i] = 1

    # TODO: Return correct class_IDs 
    # print(np.sum(mask))
    return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


def apply_mask(image, mask, color, alpha=0.5):

    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])

    return image


def create_masked_image_dataset(annotations_data):

    for img_data in annotations_data:
        # Load the image
        image = load_image(img_data["filename"], img_data["subset"])
        masked_image = image.astype(np.uint32).copy()

        masks, _ = load_mask(img_data)

        os.makedirs(
            os.path.join("../data", img_data["subset"], "masked"), exist_ok=True
        )

        N = len(img_data["polygons"])
        colors = random_colors(N)

        for i in range(N):
            color = colors[i]
            mask = masks[:, :, i]
            masked_image = masked_image + mask
            # masked_image = apply_mask(masked_image, mask, color)

        path = os.path.join("../data", img_data["subset"], "masked", img_data["filename"])
        imsave(path, masked_image)


def main():
    # "train", "test", 
    for subset in ["val"]: 
        annot = load_annotations("../data", subset)
        create_masked_image_dataset(annot)


if __name__ == "__main__":
    main()
