import os
import json
from typing import Dict, List
from collections import OrderedDict

from utils.log import Log
from utils.common_defs import class_header, method_header

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

__all__ = ['DT2ObjDetDataLoader']

log = Log()

@class_header(
    description='''
    Data loader for Detectron2 Object Detection.
    ''')
class DT2ObjDetDataLoader:
    def __init__(self, data_path: str, model_cfg: Dict, augmentations: Dict = None):

        self.data_path = data_path
        self.model_cfg = model_cfg
        self.augmentations = augmentations
        DT2ObjDetDataLoader._dataset_registration(self)
        DT2ObjDetDataLoader._write_augmentations(self)
        

    @method_header(
        description='''
        Saving Augmentations.'''
        )
    def _write_augmentations(self):
        augs_file_path = os.path.join(f"runs", f"dt",
                                     "metadata")
        os.makedirs(augs_file_path, exist_ok=True)

        if self.augmentations and 'train' in self.augmentations:
            with open(os.path.join(augs_file_path, 'train_augmentations.json'), "w") as f:
                json.dump(self.augmentations['train'], f)
        else:
            with open(os.path.join(augs_file_path, 'train_augmentations.json'), "w") as f:
                json.dump(self.model_cfg['augmentations']['train'], f)
        if self.augmentations and 'test' in self.augmentations:
            with open(os.path.join(augs_file_path, 'test_augmentations.json'), "w") as f:
                json.dump(self.augmentations['test'], f)
        else:
            with open(os.path.join(augs_file_path, 'test_augmentations.json'), "w") as f:
                json.dump(self.model_cfg['augmentations']['test'], f)


    @method_header(
        description='''
        Loading image data present in the JSON file.''',
        arguments='''
            json_path: location of the training directory
            ''' )
    def load_ann(self, data_path: str) -> Dict:
        
        for root, dirs, files in os.walk(data_path):
            for name in files:
                if name.endswith(".json"):
                    json_file_name = name

        json_file = os.path.join(data_path, json_file_name)
        with open(json_file) as f:
            imgs_anns = json.load(f)
        dataset_dicts = []
        id_counter = 0
        
        for i in imgs_anns["images"]:
            record = {}
            objs = []
            
            image_ann_id = i["id"]
            file_name = os.path.join(data_path, i["file_name"])
            height = i["height"]
            width = i["width"]

            record["file_name"] = file_name
            record["image_id"] = id_counter
            record["height"] = height
            record["width"] = width
            
            new = imgs_anns["annotations"][id_counter]
            
            image_ann_id = new["id"]
            category_id = new["category_id"]
            bbox = new["bbox"]
            iscrowd = new["iscrowd"]

            obj = {
                "id": image_ann_id,
                "image_id": id_counter,
                "category_id": category_id,
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "iscrowd": iscrowd
                }
            id_counter+=1
            objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record) 
        return dataset_dicts


    @method_header(
            description='''
            Gets the class names.''',
            arguments='''
            data_path: location of the input directory
            ''' )
    def cls_names_categories(self, data_path: str) -> List[str]:
            for root, dirs, files in os.walk(data_path):
                    for name in files:
                        if name.endswith(".json"):
                            json_file_name = name
                            break

            json_file = os.path.join(data_path, json_file_name)

            with open(json_file) as f:
                cls_categories = json.load(f)
            dct_cls_names = OrderedDict({i["id"] : i["name"] for i in cls_categories["categories"]})
        
            metadata_path = os.path.join(f"runs", f"dt", "metadata")
            os.makedirs(metadata_path, exist_ok=True)

            with open('cls_name_categories.json', 'w') as f:
                json.dump(dct_cls_names, f)

            return list(dct_cls_names.values())


    @method_header(
        description='''
        Registering the dataset.
        ''')
    def _dataset_registration(self):
      for root, dirs, files in os.walk(self.data_path):
        for name in files:
            if name.endswith(".json"):
                json_file_name = name
                break

      for d in ["train", "test"]:
        try:
            if f"dataset_{d}" in DatasetCatalog.list():
                DatasetCatalog.remove(f"dataset_{d}")
            register_coco_instances(f"dataset_{d}", {}, os.path.join(self.data_path, f"{d}", json_file_name), os.path.join(self.data_path, f"{d}"))
            MetadataCatalog.get(f"dataset_{d}").set(thing_classes= self.cls_names_categories(os.path.join(self.data_path, d)))

        except:
            if f"dataset_{d}" in DatasetCatalog.list():
                DatasetCatalog.remove(f"dataset_{d}")
            DatasetCatalog.register(f"dataset_{d}", lambda d=d: self.load_ann(os.path.join(self.data_path, d)))
            MetadataCatalog.get(f"dataset_{d}").set(thing_classes= self.cls_names_categories(os.path.join(self.data_path, d)))
      train_metadata = MetadataCatalog.get("dataset_train")
      test_metadata = MetadataCatalog.get("dataset_test")

    @method_header(
    description='''
        Helper function for creating connecting dataset path to data directory.
        ''',
    arguments='''
        path : string : Local dataset path where data is located, it should be parent directory of path and is required to be a string.
        ''')
    def config_connectors(self, path: str):

        self.connector = {}
        self.connector['trn_data_path'] = os.path.join(path, 'images')
        self.connector['trn_file_path'] = os.path.join(
            path, 'train_annotations.json')
        self.connector['test_data_path'] = os.path.join(path, 'images')
        self.connector['test_file_path'] = os.path.join(
            path, 'test_annotations.json')
        self.connector['val_data_path'] = os.path.join(path, 'images')
        self.connector['val_file_path'] = os.path.join(
            path, 'val_annotations.json')