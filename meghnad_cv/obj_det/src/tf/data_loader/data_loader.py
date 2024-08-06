import os
import sys
import json
import tempfile
from typing import Dict, Tuple

import yaml
import tensorflow as tf
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.tf.data_loader.loader_utils import get_tfrecord_dataset, get_coco_anns
from meghnad.core.cv.obj_det.src.tf.model_loader.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.tf.model_loader.utils import compute_target
from meghnad.core.cv.obj_det.src.tf.data_loader.transforms import build_transforms

__all__ = ['TFObjDetDataLoader']

log = Log()


@class_header(
    description='''
    Data loader for object detection.
    ''')
class TFObjDetDataLoader:
    def __init__(self, data_path: str, model_cfg: Dict, augmentations: Dict = None):
        self.data_path = data_path
        self.model_cfg = model_cfg

        self.batch_size = model_cfg.get('batch_size', 4)
        self.input_shape = model_cfg['input_shape'][:2]
        self.num_classes = model_cfg['num_classes']
        self.max_boxes = 100
        scales = model_cfg.get(
            'scales', [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075])

        feature_map_sizes = model_cfg.get(
            'feature_map_sizes', [38, 19, 10, 5, 3, 1])
        aspect_ratios = model_cfg.get(
            'aspect_ratios', [[2], [2, 3], [2, 3], [2, 3], [2], [2]])

        self.train_dataset = None
        self.train_size = 0
        self.validation_dataset = None
        self.val_size = 0
        self.test_dataset = None
        self.test_size = 0
        self.val_ann_file = None
        self.test_ann_file = None

        if augmentations and 'train' in augmentations:
            self.train_transforms = build_transforms(
                augmentations['train']
            )
        else:
            self.train_transforms = build_transforms(
                model_cfg['augmentations']['train'])

        if augmentations and 'test' in augmentations:
            self.test_transforms = build_transforms(
                augmentations['test']
            )
        else:
            self.test_transforms = build_transforms(
                model_cfg['augmentations']['test'])

        self.default_boxes = generate_default_boxes(
            scales, feature_map_sizes, aspect_ratios)

        self._load_data_from_directory(data_path)

    @method_header(
        description='''
            Function for data augmentation, it can be used for both training and testing configrations.
            ''',
        arguments='''
            training: boolean : Toggle to specify in which setting it should run: training or testing.
            image: tf.Tensor : it should be concerned image where augmentation will be applied, and it should strictly be a Tensor.
            bboxes: tf.Tensor : the bounding boxes of objects within image where augmentation will be applied.
            classes: tf.Tensor : The ground truth associated with each image.
            ''',
        returns='''
            a 3 member tuple containing image, bboxes and classes''')
    def _aug_fn(self, training, image: tf.Tensor, bboxes: tf.Tensor, classes: tf.Tensor) -> Tuple:
        fn = self.train_transforms if training else self.test_transforms
        data = {'image': image, 'bboxes': bboxes, 'classes': classes}
        aug_data = fn(**data)
        return aug_data['image'], aug_data['bboxes'], aug_data['classes']

    @method_header(
        description='''
            This function will prepare data in consumable format, that include decoding image, stacking multiple images using augmentation.
            ''',
        arguments='''
            tf_example: Example that will be parsed, decoded, padded, and augmented using _aug_fn function.
            training [optional]: boolean : Need training or not.
            ''',
        returns='''
            In training mode,
                image: a Tensor of image
                gt_confs: a Tensor of ground truths confidence
                gt_locs: a Tensor of ground truths locations
            In validation/test mode.
                image_id: A Tensor of image ids
                image_size: A Tensor of image size
                image: a Tensor of image
                gt_confs: a Tensor of ground truths confidence
                gt_locs: a Tensor of ground truths locations
            ''')
    def _parse_tf_example(self, tf_example: tf.train.Example, training: bool = True) -> Tuple:

        example_fmt = {
            'image/id': tf.io.FixedLenFeature([], tf.int64),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        parsed_example = tf.io.parse_single_example(tf_example, example_fmt)
        image_id = tf.cast(parsed_example['image/id'], tf.int32)
        image = tf.image.decode_jpeg(parsed_example['image/encoded'])
        image_height = tf.cast(parsed_example['image/height'], tf.int32)
        image_width = tf.cast(parsed_example['image/width'], tf.int32)
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.cast(image, tf.float32)

        xmins = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
        ymins = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
        xmaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
        ymaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
        labels = tf.cast(tf.sparse.to_dense(
            parsed_example['image/object/class/label']), tf.int32)

        tf.debugging.assert_non_positive(
            tf.reduce_sum(tf.cast(xmins > xmaxs, tf.float32)))
        tf.debugging.assert_non_positive(
            tf.reduce_sum(tf.cast(ymins > ymaxs, tf.float32)))

        bboxes = tf.stack([
            xmins,
            ymins,
            xmaxs,
            ymaxs,
        ], 1)

        # # Transformations
        image, bboxes, labels = tf.numpy_function(
            func=self._aug_fn,
            inp=[training, image, bboxes, labels],
            Tout=[tf.float32, tf.float32, tf.int32])

        # Pad
        num_pad = tf.maximum(0, self.max_boxes - tf.shape(labels)[0])
        bboxes = tf.pad(bboxes, [[0, num_pad], [0, 0]])
        labels = tf.pad(labels, [[0, num_pad]])
        bboxes = tf.reshape(bboxes, [self.max_boxes, 4])
        labels = tf.reshape(labels, [self.max_boxes])

        # Recover shapes
        image.set_shape((self.input_shape[0], self.input_shape[1], 3))
        bboxes.set_shape((self.max_boxes, 4))
        labels.set_shape((self.max_boxes,))

        # Compute targets
        gt_confs, gt_locs = compute_target(
            self.default_boxes, bboxes, labels)
        if training:
            return image, gt_confs, gt_locs
        else:
            return image_id, tf.stack([image_height, image_width]), image, gt_confs, gt_locs

    @method_header(
        description='''
            Helper function for loading data from directory, distributing it into training, testing, and validation set.
            ''',
        arguments='''
            path : The path where data is located, as of now only JSON format is supported.
            ''',
        returns='''
            A tuple of training/validation/test TFDataset instances.''')
    def _load_data_from_directory(self, path: str) -> Tuple:
        autotune = tf.data.AUTOTUNE
        train_dataset, self.train_size = self._read_data(path, 'train')

        with open(path) as f:
            data_dict = yaml.safe_load(f)
            names = data_dict['names']
            self.class_map = {i + 1: name for i, name in enumerate(names)}

        train_dataset = train_dataset.shuffle(8 * self.batch_size)
        train_dataset = train_dataset.map(
            lambda x: self._parse_tf_example(x, True), num_parallel_calls=autotune,
        )
        train_dataset = train_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 0, 0.0), drop_remainder=True
        )
        train_dataset = train_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.train_dataset = train_dataset.prefetch(autotune)

        validation_dataset, self.val_size = self._read_data(path, 'val')
        validation_dataset = validation_dataset.map(
            lambda x: self._parse_tf_example(x, False), num_parallel_calls=autotune,
        )
        validation_dataset = validation_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0, 0, 0.0, 0, 0.0), drop_remainder=True
        )
        validation_dataset = validation_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.validation_dataset = validation_dataset.prefetch(autotune)

        test_dataset, self.test_size = self._read_data(path, 'test')
        test_dataset = test_dataset.map(
            lambda x: self._parse_tf_example(x, False), num_parallel_calls=autotune,
        )
        test_dataset = test_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0, 0, 0.0, 0, 0.0), drop_remainder=True
        )
        test_dataset = test_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.test_dataset = test_dataset.prefetch(autotune)
        return self.train_dataset, self.validation_dataset, self.test_dataset

    @method_header(
        description='''
            Helper function for creating connecting dataset path to data directory.
            ''',
        arguments='''
            data_file: path to YOLO data file.
            dataset_split: string : which dataset to choose (train) is selected by default
            ''',
        returns='''
            returns dataset and number of samples in the form of tensor records''')
    def _read_data(self, data_file: str, dataset_split: str) -> Tuple:
        tfrecord_dir = '.'  # TODO:
        config_name = self.model_cfg['config_name']
        tfrecord_file = os.path.join(
            tfrecord_dir, f'{config_name}_{dataset_split}.tfrecord')
        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    f'tfrecord_file: {tfrecord_file}')
        dataset, num_samples = get_tfrecord_dataset(
            data_file, dataset_split, tfrecord_file
        )

        # For validation
        if dataset_split in ('val', 'test'):
            anns = get_coco_anns(data_file, dataset_split)
            tmp_file = tempfile.NamedTemporaryFile(
                prefix=config_name, suffix='.json').name

            with open(tmp_file, 'wt') as f:
                json.dump(anns, f)

            if dataset_split == 'val':
                self.val_ann_file = tmp_file
            elif dataset_split == 'test':
                self.test_ann_file = tmp_file
        return dataset, num_samples
