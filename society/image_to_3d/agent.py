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


class HumanReconstruction:
    def __init__(self, device):
        print(f"Initializing HRNet to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/cv_hrnet_image-human-reconstruction'
        self.human_reconstruction_pipeline = pipeline(Tasks.human_reconstruction, model=model_id)

    @prompts(name="HumanReconstruction",
             description="Useful when you turn a personal photo into 3D mesh. Receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):

        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        image_path = image_path.strip("\n")
        result =  self.human_reconstruction_pipeline(image_path)
        #cv2.imwrite(image_filename, result[OutputKeys.OUTPUT_IMG])
        mesh = result[OutputKeys.OUTPUT]
        return mesh
    
if __name__ == "__main__":
    hr_model = HumanReconstruction(device="cuda:0")
    docs = hr_model.inference("WechatIMG899.jpeg")
    print(docs)