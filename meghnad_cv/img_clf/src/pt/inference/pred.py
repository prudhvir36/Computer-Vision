import os
from PIL import Image
import json
import torch
import torchvision
import collections

from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.img_clf.cfg.img_clf_config import ImgClfConfig


log = Log()


@class_header(
    description='''
    Image classification prediction
    ''',
    )
class PTImgClfPred:
    def __init__(self,
                 classes: list,
                 test_path:str,
                 output_path:str,
                 device:str='cpu') -> None:
        self.test_path=test_path
        self.output_path=output_path
        self.classes=classes
        self.model = torch.jit.load(os.path.join(self.output_path,'metadata.pt'))
        self.device=device
        self.ImgClfConfig=ImgClfConfig(MeghnadConfig)


    @method_header(
        description='''Test the model trained on custom data''',
        returns='''
        predicted_class: Predicated class is written in dictionary
        ''')
    def pred(self) -> dict:
        image_data={}
        with open(os.path.join(self.output_path, 'augmenations.json'), 'r') as json_file:
            augmenations = json.load(json_file)
            augmenations=augmenations['train']
        for filename in os.listdir(self.test_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(os.path.join(self.test_path, filename))
                img_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((augmenations['resize']['width'], augmenations['resize']['height'])),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        augmenations['normalize']['mean'],
                        augmenations['normalize']['std']
                    )])
                img_tensor = img_transforms(img).unsqueeze(0)
                if self.ImgClfConfig.get_meghnad_configs('DEVICE') != 'cpu' \
                        and self.ImgClfConfig.get_meghnad_configs('DEVICE') != None:
                    if torch.cuda.is_available():
                        self.device = self.configs.get_meghnad_configs('DEVICE')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                img_tensor = img_tensor.to(device)
                self.model.eval()
                with torch.no_grad():
                    output = self.model(img_tensor)
                class_names = self.classes
                predicted_index = torch.argmax(output[0])
                predicted_class = class_names[predicted_index]
                image_data[f'{filename}'] = predicted_class

        value_count = collections.Counter(image_data.values())
        count_per_class={'count':value_count}
        with open(os.path.join(self.output_path, 'prediction.json'), 'w') as f:
            json.dump(image_data, f)
            json.dump(count_per_class,f)
        return predicted_class


