import sys
sys.path.append("..")

import os
import cv2
import json
import torch
import random
import requests
import numpy as np
from typing import List

from utils.log import Log
from utils import ret_values
from meghnad.cfg.config import MeghnadConfig
from utils.common_defs import class_header, method_header
from meghnad.core.cv.zero_shot_img_seg.cfg.zsis_config import ZeroShotImageSegmentationConfig

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

log = Log()

_all_ = ["ZeroShotImageSegmentation"]

@class_header(
description='''
Zero Shot Image Segmentation.
''')
class ZeroShotImageSegmentation:

    def __init__(self, model_type: List[str] = ["default"], device:str = "cpu") -> None:
        self.model_type, = model_type
        self.device = device
        self.zsiscfg = ZeroShotImageSegmentationConfig(MeghnadConfig())
        self.save_dwn_model = MeghnadConfig().get_meghnad_configs('INT_PATH')
        self.model_name = self.zsiscfg.get_model_cfg(self.model_type)["model_type"]

        if self.zsiscfg.get_meghnad_configs('DEVICE') != 'cpu' and self.zsiscfg.get_meghnad_configs('DEVICE') != None:
            if torch.cuda.is_available():
                self.device = self.zsiscfg.get_meghnad_configs('DEVICE')

        log.STATUS(sys._getframe().f_lineno,
                        __file__, __name__, 
                        f"Loading Segmentation Model with {self.model_type} Setting")
        self.model_path = self._select_model()
        self.model = sam_model_registry[self.model_name](checkpoint=self.model_path).to(self.device)
    

    @method_header(
            description="""Selecting Model based on the Model type specified.""")
    def _select_model(self) -> str:
        model_url = self.zsiscfg.get_model_cfg(self.model_type)["model_url"]
        save_path = self._download_model(url = model_url)
        return save_path
    

    @method_header(
            description="""Downloading Model based on the Model type specified.""",
            arguments="""
            url: model url to download the specified model""")
    def _download_model(self, url: str) -> None:
        save_path = os.path.join(self.save_dwn_model, "model.pth")
        try:
            response = requests.get(url)
        
            if response.status_code != 200:
                log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__, 
                        f"Failed to download model from {url}. Response code: {response.status_code}")
        
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            return save_path
        
        except Exception as e:
            log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__, 
                        f"Error downloading model from {url}: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)

    @method_header(description="""
                   Takes in the image and generates masks with bbox co-ordinates.""",
                   arguments="""
                   image_path: Input image path
                   filename: str - Image filename
                   bbox: List[List[str]] - Bounding box coordinates
                   output_path: str - Output image path
                   """)
    def _img_wrt_bbox(self, image_path: str, filename: str, 
                      bbox: List[List[float]], output_path: str) -> None:
        bbox_overlay = bbox
        bbox = np.array(bbox)
        read_image = cv2.imread(image_path)
        image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)

        sam_predictor = SamPredictor(self.model)
        sam_predictor.set_image(image)
        
        result_masks = []
        for box in bbox:
            masks, scores, logits = sam_predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        
        mask_arr_lst = []
        for i in result_masks:
            mask_arr_lst.append(np.array(i, dtype=np.uint8))
            
        self._write_segmentations(mask_arr_lst, filename=filename, output_path=output_path)
        
        output_image_path = os.path.join(output_path, f"seg_{filename}")
        self._overlay_masks(image = read_image,
                            bbox_coor=bbox_overlay,
                    mask_arrays = mask_arr_lst,
                    output_image_path = output_image_path)
        
    @method_header(description="""
                   Takes in the image and generates masks.""",
                   arguments=""""
                   image_path: Input image path
                   filename: str - Image filename
                   output_path: str - Output image path
                   iou_thresh: float - Takes in a float value
                   """)
    def _img_without_bbox(self, image_path: str, filename: str, 
                          output_path: str, iou_thresh: float) -> None:

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=iou_thresh,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100)
                        
        masks = mask_generator.generate(image)
        self._save_masked_image(masks, os.path.join(output_path, f"seg_{filename}"))
        self._write_segmentations(masks, filename=filename, output_path=output_path)

    @method_header(description="""
                   Takes in the generated masks and filename the image needs to be saved.""",
                   arguments=""""
                   anns: masks - Takes in the generated masks as the input
                   filename - Image name that needs to be saved
                   """)
    def _save_masked_image(self, anns: np.array, filename: str) -> None:
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(filename, img_bgr)

    @method_header(description="""
                   Overlays Mask on Input Image.""",
                   arguments=""""
                   image: np.ndarray - Takes in np.array of the input image
                   mask_arrays: np.ndarray - Takes in the masks that needs to be overlaid on the input image
                   filename - Image name that needs to be saved
                   output_path - Path to the output directory
                   """)
    def _overlay_masks(self, image: np.array, mask_arrays: np.ndarray, 
                       output_image_path: str, bbox_coor = None) -> None:
        input_image = image

        resized_masks = []
        for mask_array in mask_arrays:
            resized_mask = cv2.resize(mask_array, (input_image.shape[1], input_image.shape[0]))
            resized_masks.append(resized_mask)
            
        colors = [(random.randint(0, 255), 
                   random.randint(0, 255),
                     random.randint(0, 255)) 
                     for i in range(len(mask_arrays))]
            
        output_image = np.copy(input_image)
        for mask, color in zip(resized_masks, colors):
            color_overlay = np.zeros_like(input_image)
            color_overlay[np.where(mask > 0)] = color

            outputs_images = cv2.addWeighted(output_image, 1, color_overlay, 0.9, 0)

        if not bbox_coor == None:
            bboxes = bbox_coor
            for bbox in bboxes:
                x1, y1, w, h = [int(c) for c in bbox]
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(outputs_images, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imwrite(output_image_path, outputs_images)


    @method_header(description="""
                   Saving Segmentations generated by the Model.""",
                   arguments=""""
                   masks: masks - Takes in the generated masks as the input
                   filename - Image name that needs to be saved
                   output_path - Path to the output directory
                   """)
    def _write_segmentations(self, masks: np.array, filename: str, output_path:str) -> None:
        seg_lst = []
        seg_dct = {}
        for i, mask in enumerate(masks):
            try:
                binary_mask = mask.squeeze().astype(np.uint8)
            except:
                binary_mask = masks[i]['segmentation'].squeeze().astype(np.uint8)
                
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
            segmentation = largest_contour.flatten().tolist()
            seg_lst.append(segmentation)
        seg_dct[filename] = seg_lst
        json_obj = json.dumps(seg_dct)
        
        with open(os.path.join(output_path, "segmentation_results.json"), 'a') as f:
            f.write(json_obj)


    @method_header(description="""Takes Directory as an input and return Image Captions""",
                   arguments="""
                   input_path: str = Directory as an input,
                   bbox_json: str = JSON path of the input box coordinates of images
                   output_path: str = Output Directory,
                   iou_thresh: float = IOU threshold for the image segmentations
                   """)
    def pred(self, input_path: str, output_path: str = None,
              bbox_json: str = None, iou_thresh: float = 0.86) -> None:
        
        if output_path == None:
            log.ERROR(sys._getframe().f_lineno,
                            __file__, __name__, 
                            "[!] Mention Output Path to Save Results")
            return ret_values.IXO_RET_INVALID_INPUTS
        
        os.makedirs(output_path, exist_ok=True)

        if not bbox_json == None:
            json_file_path = bbox_json
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
        else:
            json_data = {}

        seg_result = os.path.join(output_path, "segmentation_results.json")
        if os.path.exists(seg_result):
            os.remove(seg_result)

        for root, dirs, files in os.walk(input_path):
            for name in files:
                if name.endswith((".jpg", ".jpeg", ".png")):
                    image_file = name
                    image_path = os.path.join(root, image_file)
                    
                    if image_file in json_data:
                        bbox = json_data[image_file]
                        self._img_wrt_bbox(image_path, image_file, json_data[image_file], output_path)
                    else:
                        self._img_without_bbox(image_path, image_file, output_path, iou_thresh)    

                    if self.device != 'cpu':
                      torch.cuda.empty_cache()
        return ret_values.IXO_RET_SUCCESS