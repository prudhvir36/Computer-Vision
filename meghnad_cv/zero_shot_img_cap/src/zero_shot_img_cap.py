import os
import sys
import PIL
import torch
import requests
from PIL import Image
from typing import Any, List, Tuple

from utils.log import Log
from utils import ret_values
from utils.common_defs import class_header, method_header
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.zero_shot_img_cap.cfg.img_cap_cfg import ZeroShotImageCaptioningConfig

from transformers import BlipProcessor, BlipForConditionalGeneration

log = Log()

_all_ = ["ZeroShotImageCaptioning"]

@class_header(
description='''
Zero-shot Image Captioning.
''')
class ZeroShotImageCaptioning:

    def __init__(self, model_type: List[str] = ["default"], device:str = "cpu") -> None:
        self.model_type, = model_type
        self.device = device
        self.zcfg = ZeroShotImageCaptioningConfig(MeghnadConfig())
        self.processor = BlipProcessor.from_pretrained(self._select_model())        
        if self.zcfg.get_meghnad_configs('DEVICE') != 'cpu' and self.zcfg.get_meghnad_configs('DEVICE') != None:
            if torch.cuda.is_available():
                self.device = self.zcfg.get_meghnad_configs('DEVICE')

        log.STATUS(sys._getframe().f_lineno,
                        __file__, __name__, 
                        "Loading Blip Model with {self.model_type} Setting")
        self.model = BlipForConditionalGeneration.from_pretrained(self._select_model()).to(self.device)
        

    @method_header(
            description="""Selecting Model based on the Model type specified.""")
    def _select_model(self):
      return self.zcfg.get_model_cfg(self.model_type)["repo_id"]


    @method_header(description="""Takes image as an Input and preprocess it""",
                   arguments="""
                   input: Any = Image should be passed as an Input parameter""",
                   returns="""PIL Image Object
                   """)
    def _load_image(self, input: Any) -> Tuple:
        
        if isinstance(input, str):
            if input.startswith("http://") or input.startswith("https://") or input.startswith("www."):
                raw_image = Image.open(requests.get(input, stream=True).raw).convert('RGB')
            elif os.path.isfile(input):
                raw_image = PIL.Image.open(input).convert('RGB')
            else:
                log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__, "Incorrect URL path")
                return ret_values.IXO_RET_INVALID_INPUTS, input
                
        elif isinstance(input, PIL.Image.Image):
            raw_image = input.convert('RGB')
        else:
            log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__, "[!] Incorrect format of Image. Should be a URL or Directory of Input Image")
            return ret_values.IXO_RET_INVALID_INPUTS, input
        return ret_values.IXO_RET_SUCCESS, raw_image

    @method_header(description="""Takes Directory as an input and return Image Captions""",
                   arguments="""
                   input_path: str = Directory as an input,
                   output: str = Output Directory""",
                   returns="""list of Image Captions
                   """)
    def pred(self, input_path: str, output_path:str ) -> List:

        os.makedirs(output_path, exist_ok=True)
        img_captions_lst = []
        file_counter = 0

        for root, dirs, files in os.walk(input_path):
            for name in files:
                if name.endswith((".jpg", ".jpeg", ".png")):
                    image_file = name

                    ret_value, image = self._load_image(os.path.join(root, image_file))
                    inputs = self.processor(image, return_tensors="pt").to(self.device)

                    out = self.model.generate(**inputs)
                    img_caption = self.processor.decode(out[0], skip_special_tokens=True)

                    if self.device != 'cpu':
                      torch.cuda.empty_cache()

                    img_captions_lst.append(f"{file_counter}. {img_caption}")
                    file_counter+=1

        with open(os.path.join(output_path, "captions.txt"), 'w') as write_results:
            for caption in img_captions_lst:
                write_results.write(caption+"\n")

        return img_captions_lst

