# Lint as: python3
'''Extract the MS COCO dataset into person/not person training dataset.

Extract and segment the MS COCO dataset into person and no person image sets of
equal size. Save images in dataset/person and dataset/no_person subsets ready
for training.

The 2017 train images and train/val annotations must be downloaded from
https://cocodataset.org/#download and extracted before this script can be run.
The images should be extracated in the same directory as this script.
'''
import glob
import json
import os

from absl import app

from PIL import Image
from resizeimage import resizeimage

IMAGE_DIMS = 128
OUTPUT_DIR = 'dataset'


def write_image(image_id, annotation_file_name, is_person):
    image_name = "{}.jpg".format(str(image_id).zfill(12))
    image_dir = annotation_file_name.split('_')[1].split('.')[0] + '/'
    image_path = os.path.join(image_dir, image_name)

    with Image.open(image_path) as image:
        cover = resizeimage.resize_cover(image, [IMAGE_DIMS, IMAGE_DIMS])
        if is_person:
            save_file_name = "{}/person/{}".format(OUTPUT_DIR, image_name)
        else:
            save_file_name = "{}/no_person/{}".format(OUTPUT_DIR, image_name)
        cover.save(save_file_name, image.format)

def ensure_image_quality(annotation, image):
    bounding_box = annotation['bbox']
    if annotation['iscrowd'] != 0:
        return False

    width = image['width']
    height = image['height']
    min_dimension = min(width, height)

    # Ensure person occupies 20% of image area
    if bounding_box[2] * bounding_box[3] < IMAGE_DIMS * IMAGE_DIMS / 5:
        return False

    # Ensure bounding box will not get cropped out.
    if width > height:
        crop_pixels = (width - min_dimension) / 2 # symmetric cropping
        if (bounding_box[0] <= crop_pixels
                or (bounding_box[0] + bounding_box[2]) >= (width - crop_pixels)):
            return False
    else:
        crop_pixels = (height - min_dimension) / 2 # symmetric cropping
        if (bounding_box[1] <= crop_pixels
                or (bounding_box[1] + bounding_box[3]) >= (height - crop_pixels)):
            return False
    return True


def can_write_image(image):
    return image['width'] >= IMAGE_DIMS and image['height'] >= IMAGE_DIMS

def configure_dataset_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    person_dir = os.path.join(OUTPUT_DIR, 'person')
    if not os.path.exists(person_dir):
        os.mkdir(person_dir)

    no_person_dir = os.path.join(OUTPUT_DIR, 'no_person')
    if not os.path.exists(no_person_dir):
        os.mkdir(no_person_dir)

    # Clear dataset directory
    for fname in glob.glob(os.path.join(person_dir, '*')):
        os.remove(fname)
    for fname in glob.glob(os.path.join(no_person_dir, '*')):
        os.remove(fname)


# Fetch person category id.
def get_person_id(data):
    categories = data['categories']
    person_id = -1
    for category in categories:
        if category['name'] == 'person':
            person_id = category['id']
    return person_id


# Populate ditionary mapping each image to a list of annotations.
def generate_annotations_dict(data):
    annotations = data['annotations']
    images = data['images']

    annotations_dict = {}
    for image in images:
        annotations_dict.update({image['id'] : []})

    for annotation in annotations:
        annotations_dict[annotation['image_id']].append(annotation)
    return annotations_dict


# Populate dictionary mapping image id to image data.
def generate_image_dict(data):
    images = data['images']
    image_dict = {}
    for image in images:
        image_dict.update({image['id'] : image})
    return image_dict


def main(argv):
    if len(argv) != 2:
        raise app.UsageError('Usage: parse_coco.py <path to instances_train2017.json>')

    data = json.load(open(argv[1]))
    print("loaded data")

    images = data['images']

    person_id = get_person_id(data)
    annotations_dict = generate_annotations_dict(data)
    image_dict = generate_image_dict(data)

    # Create dirs for dataset
    configure_dataset_directory()

    # Walk through annotations, copying best images of people to positive dataset.
    print('generating positive examples')
    num_person = 0
    for image in images:
        image_id = image['id']
        for annotation in annotations_dict[image_id]:
            is_person = annotation['category_id'] == person_id

            image = image_dict[image_id]
            useful_person_image = is_person and ensure_image_quality(annotation, image)

            if useful_person_image and can_write_image(image):
                write_image(image_id, argv[1], True)
                num_person += 1
                if num_person % 100 == 0:
                    print('.', end='', flush=True)
                break

    # Walk through remaining annotations, copying no-person image to negative set.
    print('\ngenerating negative examples')
    num_no_person = 0
    for image in images:
        image_id = image['id']
        has_person = False
        for annotation in annotations_dict[image_id]:
            has_person = has_person or annotation['category_id'] == person_id

        if not has_person and can_write_image(image):
            write_image(image_id, argv[1], False)
            num_no_person += 1
            if num_no_person % 100 == 0:
                print('.', end='', flush=True)

        # Only save as many negative sample as we have positive.
        if num_no_person >= num_person:
            break


if __name__ == '__main__':
    app.run(main)
