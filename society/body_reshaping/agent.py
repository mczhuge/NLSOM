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


class SAFG:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_id = 'damo/cv_flow-based-body-reshaping_damo'
        self.pipeline_image_body_reshaping = pipeline(Tasks.image_body_reshaping, model=model_id)

    @prompts(name="SAFG",
             description="Useful when you want to make the body in the photo more beautiful. Receives image_path as input."
                         "Applications involving scenes that require body contouring."
                         "The input to this tool should be a string, representing the image_path.")
    def inference(self, image_path):
        
        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        #image_filename = f"{str(uuid.uuid4())[:8]}.png"
        image_path = image_path.strip("\n")
        result =  self.pipeline_image_body_reshaping(image_path)
        cv2.imwrite(image_filename, result[OutputKeys.OUTPUT_IMG])
        return image_filename
    

if __name__ == "__main__":
    skin_touching_model = SAFG(device="cuda:0")
    image = skin_touching_model.inference("d317f96a.png")