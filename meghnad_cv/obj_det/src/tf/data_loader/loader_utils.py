import os
from typing import List, Tuple, Dict

import cv2
import yaml
import numpy as np
import tensorflow as tf
import tqdm

from utils.log import Log
from utils.common_defs import method_header
from meghnad.core.cv.obj_det.src.utils.general import get_sync_dir

log = Log()


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value: List[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(value: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


@method_header(
    description='''
    Build record for training or testing.
    ''',
    arguments='''
    image_info : dict :information regarding the image i.e file_name etc
        ''',
    returns='''
    a tensor with all image information''')
def _build_record(image_info: Dict) -> tf.train.Example:

    image_path = image_info['filename']
    image = open(image_path, 'rb').read()

    ext = image_info['filename'].split('.')[-1]
    if ext.lower() in ('jpg', 'jpeg'):
        ext = b'jpg'
    elif ext.lower() == 'png':
        ext = b'png'

    bboxes = np.array(image_info['bboxes'])
    xmins = bboxes[..., 0].tolist()
    xmaxs = bboxes[..., 1].tolist()
    ymins = bboxes[..., 2].tolist()
    ymaxs = bboxes[..., 3].tolist()
    class_names = [name.encode('utf-8') for name in image_info['class_names']]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/id': _int64_feature(image_info['id']),
        'image/height': _int64_feature(image_info['height']),
        'image/width': _int64_feature(image_info['width']),
        'image/filename': _bytes_feature(image_info['filename'].encode('utf-8')),
        'image/encoded': _bytes_feature(image),
        'image/format': _bytes_feature(ext),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(class_names),
        'image/object/class/label': _int64_list_feature(image_info['classes']),
    }))
    return tf_example

@method_header(
    description='''
    Load YOLO annotations from given file.
    ''',
    arguments='''
        text_file: The annotation file.
        classes: A list of class names.
    ''',
    returns='''
        A 3-tuple of bounding boxes, class ids and class names.
        bboxes: A list of bounding boxes [cx, cy, w, h]
        class_ids: A list of corresponding class ids.
        class_names: A list of corresponding class names.
    '''
)
def _load_yolo_ann(text_file: str, classes: List[str]) -> Tuple:
    bboxes = []
    class_ids = []
    class_names = []
    with open(text_file) as f:
        for line in f:
            line = line.strip()
            row = list(map(float, line.split(' ')))
            class_id = int(row[0])
            cx, cy, w, h = row[1:]
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)

            bboxes.append([x1, y1, w, h])
            class_ids.append(class_id + 1)
            class_names.append(classes[class_id])
    return bboxes, class_ids, class_names


@method_header(
    description='''
    Load YOLO dataset from given images directory.
    ''',
    arguments='''
        image_dir: Images directory.
        classes: A list of class names.
    ''',
    returns='''
        data: A mapping from image_id to its info
    '''
)
def _load_yolo_dataset(image_dir: str, classes: List[str]) -> Dict:
    label_dir = image_dir.replace('images', 'labels')
    filenames = os.listdir(image_dir)
    data = dict()
    for i, filename in enumerate(filenames):
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        label_path = os.path.join(label_dir, filename.split('.')[0] + '.txt')
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            continue
        bboxes, class_ids, class_names = _load_yolo_ann(
            label_path, classes)
        image_id = i + 1
        data[image_id] = {
            'id': image_id,
            'width': img_h,
            'height': img_w,
            'filename': image_path,
            'bboxes': bboxes,
            'classes': class_ids,
            'class_names': class_names
        }
    return data


@method_header(
    description='''
    method to get records in form of tensorflow dataset.
    ''',
    arguments='''
    image_dir: directory where the test/train images are present
    ann_file: path where the annotation file is present
        ''',
    returns='''
    returns dataset in form of tensor and number_of_samples in int''')
def get_tfrecord_dataset(data_file: str, dataset_split: str, tfrecord_file: str) -> Tuple:
    with open(data_file) as f:
        data_dict = yaml.safe_load(f)

    if dataset_split == 'test' and data_dict.get('test') is None:
        dataset_split = 'val'
    sync_dir = get_sync_dir()
    path = data_dict[dataset_split]
    path = os.path.join(sync_dir, path)
    names = data_dict['names']
    if isinstance(names, dict):
        names = list(names.values())

    dataset_dict = _load_yolo_dataset(path, names)
    if not os.path.isfile(tfrecord_file):
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for _, image_info in tqdm.tqdm(dataset_dict.items()):
                example = _build_record(image_info)
                writer.write(example.SerializeToString())

    # Initialize a dataset from the above tf record file
    num_samples = len(dataset_dict)
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    return dataset, num_samples


@method_header(
    description='''
    Load COCO annotations from given file.
    ''',
    arguments='''
        data_file: COCO annotation file.
        dataset_split: Split name (train/val/test)
    ''',
    returns='''
        data: A mapping with the following structure.
        {
            "images": [{"id": 12, "width": 500, "height": 1000, "file_name": "1.jpg"}, ...],
            "annotations": [{"bbox": [x1, y1, w, h], "area": w*h, "iscrowd": 0, "category_id": 0, "id": 100, "image_id": 1}, ...]
        }
    '''
)
def get_coco_anns(data_file: str, dataset_split: str) -> Dict:
    with open(data_file) as f:
        data_dict = yaml.safe_load(f)

    if dataset_split == 'test' and data_dict.get('test') is None:
        dataset_split = 'val'
    image_dir = data_dict[dataset_split]
    names = data_dict['names']
    if isinstance(names, dict):
        names = list(names.values())

    categories = [{'supercategory': name, 'id': i + 1, 'name': name}
                  for i, name in enumerate(names)]
    label_dir = image_dir.replace('images', 'labels')
    sync_dir = get_sync_dir()
    image_dir = os.path.join(sync_dir, image_dir)
    label_dir = os.path.join(sync_dir, label_dir)
    filenames = os.listdir(image_dir)
    data = dict(
        images=[],
        annotations=[],
        categories=categories
    )
    ann_id = 0
    for i, filename in enumerate(filenames):
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        label_path = os.path.join(label_dir, filename.split('.')[0] + '.txt')
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            continue
        bboxes, class_ids, class_names = _load_yolo_ann(
            label_path, names)
        image_id = i + 1
        data['images'].append({
            'id': image_id,
            'width': img_h,
            'height': img_w,
            'file_name': filename,
        })

        for bbox, class_id in zip(bboxes, class_ids):
            ann_id += 1
            x, y, w, h = bbox
            x = int(x * img_w)
            y = int(y * img_h)
            w = int(w * img_w)
            h = int(h * img_h)
            data['annotations'].append({
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'category_id': class_id,
                'id': ann_id,
                'image_id': image_id
            })
    return data
