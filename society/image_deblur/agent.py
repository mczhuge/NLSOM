from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import torch
from PIL import Image
import os
import uuid
import cv2

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator




class NAFNet:
    def __init__(self, device):
        print(f"Initializing NAFNet to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/cv_nafnet_image-deblur_reds'
        self.image_deblur_pipeline = pipeline(Tasks.image_deblurring, model=model_id)

    @prompts(name="NAFNet (Image Deblur)",
             description="Useful when you turn a blurry photo into a clear one. Receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):

        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        image_path = image_path.strip("\n")
        result =  self.image_deblur_pipeline(image_path)
        cv2.imwrite(image_filename, result[OutputKeys.OUTPUT_IMG])
        return image_filename
    
if __name__ == "__main__":
    color_model = NAFNet(device="cuda:0")
    docs = color_model.inference("blurry.jpg")
    print(docs)