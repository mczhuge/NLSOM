from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import torch
import os
import uuid
import cv2

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class ABPN:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/cv_unet_skin-retouching'
        self.pipeline_skin_retouching = pipeline(Tasks.skin_retouching, model=model_id)

    @prompts(name="ABPN",
             description="Useful when you want to make the face in the photo more beautiful. Receives image_path as input."
                         "Applications involving skin beautification, such as photo retouching."
                         "The input to this tool should be a string, representing the image_path.")
    def inference(self, image_path):
        
        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        #image_filename = "result.png"
        image_path = image_path.strip("\n")
        result =  self.pipeline_skin_retouching(image_path)
        cv2.imwrite(image_filename, result[OutputKeys.OUTPUT_IMG])
        return image_filename
    

if __name__ == "__main__":
    skin_touching_model = ABPN(device="cuda:0")
    image = skin_touching_model.inference("xyz456.png")