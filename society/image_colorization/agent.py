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

class DDColor:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/cv_ddcolor_image-colorization'
        self.pipeline_colorization = pipeline(Tasks.image_colorization, model=model_id)

    @prompts(name="DDColor",
             description="Useful when you make a gray image into color. Receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):

        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        #image_filename = "result.png"
        image_path = image_path.strip("\n")
        result =  self.pipeline_colorization(image_path)
        cv2.imwrite(image_filename, result[OutputKeys.OUTPUT_IMG])
        return image_filename
    

# class UNet:
#     def __init__(self, device):
#         print(f"Initializing UNet to {device}")
#         self.device = device
#         self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
#         model_id = 'damo/cv_unet_image-colorization'
#         self.pipeline_colorization = pipeline(Tasks.image_colorization, model=model_id)

#     @prompts(name="UNet (Image Colorization)",
#              description="Useful when you make a gray image into color. Receives image_path as input. "
#                          "The input to this tool should be a string, representing the image_path. ")
#     def inference(self, image_path):

#         image_filename = f"{str(uuid.uuid4())[:8]}.png"#os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         image_path = image_path.strip("\n")
#         result =  self.pipeline_colorization(image_path)
#         cv2.imwrite(image_filename, result['output_img'])
#         return image_filename
    
if __name__ == "__main__":
    color_model = DDColor(device="cuda:0")
    img = color_model.inference("xyz321.jpeg")
    print(img)