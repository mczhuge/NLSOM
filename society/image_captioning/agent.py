from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class OFA_large_captioning:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/ofa_image-caption_coco_large_en'
        self.pipeline_caption = pipeline(Tasks.image_captioning, model=model_id, model_revision='v1.0.1')

    @prompts(name="OFA_large_captioning",
             description="Useful when you want to know what is inside the photo. Receives image_path as input. "
                         "If there are other captioning methods, it is also suggested to utilize other captioning methods to better know the image."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        image_path = image_path.strip("\n")
        captions = self.pipeline_caption(image_path)[OutputKeys.CAPTION]
        return captions[0]


if __name__ == "__main__":
    ic = BLIP2_captioning("cuda:0")
    desc = ic.inference("d317f96a.png")

# You can add these 3 candidates
"""
class mPLUG_captioning:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/mplug_image-captioning_coco_large_en'
        self.pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)

    @prompts(name="mPLUG_captioning",
             description="Useful when you want to know what is inside the photo. Receives image_path as input. "
                         "If there are other captioning methods, it is also suggested to utilize other captioning methods to better know the image."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        image_path = image_path.strip("\n")
        captions = self.pipeline_caption(image_path)
        print(f"\nProcessed mPLUG_captioning, Input Image: {image_path}, Output Text: {captions}")
        return captions["caption"]


class OFA_distilled_captioning:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/ofa_image-caption_coco_large_en'
        self.pipeline_caption = pipeline(Tasks.image_captioning, model=model_id, model_revision='v1.0.1')

    @prompts(name="OFA_distilled_captioning",
             description="Useful when you want to know what is inside the photo. Receives image_path as input. "
                         "If there are other captioning methods, it is also suggested to utilize other captioning methods to better know the image."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        image_path = image_path.strip("\n")
        captions = self.pipeline_caption(image_path)[OutputKeys.CAPTION]
        return captions

class BLIP2_captioning:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_id = 'Salesforce/blip2-flan-t5-xl'
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)

    @prompts(name="BLIP2_captioning",
             description="Useful when you want to know what is inside the photo. Receives image_path as input."
                         "If there are other captioning methods, it is also suggested to utilize other captioning methods to better know the image."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, input):

        raw_image = Image.open(input).convert("RGB")
        inputs = self.processor(images=raw_image, return_tensors="pt").to("cuda", torch.float16)
        generated_answer_ids = self.BLIP2_MODEL.generate(**inputs)
        answer = self.processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()

        return answer
"""