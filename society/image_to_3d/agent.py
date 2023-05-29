from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os
import uuid
import urllib.request
import shutil


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class HRNet:
    def __init__(self, device="cuda:0"):
        model_id = 'damo/cv_hrnet_image-human-reconstruction'
        self.human_reconstruction_pipeline = pipeline(Tasks.human_reconstruction, model=model_id)

    @prompts(name="HRNet",
             description="Useful when you turn a personal photo into 3D mesh. Receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):

        color_filename = os.path.join('data', f"human_color_{str(uuid.uuid4())[:8]}.obj")
        recon_filename = os.path.join('data', f"human_reconstruction_{str(uuid.uuid4())[:8]}.obj")
        image_path = image_path.strip("\n")
        result =  self.human_reconstruction_pipeline(image_path)
        mesh = result[OutputKeys.OUTPUT]
        shutil.move("human_color.obj", color_filename)
        shutil.move("human_reconstruction.obj", recon_filename)
        return recon_filename
    
if __name__ == "__main__":
    hr_model = HRNet(device="cuda:0")
    docs = hr_model.inference("data/WechatIMG899.jpeg")
    print(docs)