import uuid
import os
import io
import torch
from PIL import Image
import requests

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class DETR:
    def __init__(self, device="cpu"):
        self.device = device
        self.API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
        self.headers = {"Authorization": "Bearer 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"}

    @prompts(name="DETR (object detection)",
             description="useful when you want to detect the objects in an image. "
                         "The input to this tool should be a string, representing input image file. ")
    def inference(self, filename):
        
        output = self.query(filename)
        return output

    
    def query(self, filename):
            
        with open(filename, "rb") as f:
            data = f.read()

        response = requests.post(self.API_URL, headers=self.headers, data=data)
        
        return response.json()


if __name__ == "__main__":
    object_detection_model = DETR(device="cpu")
    result = object_detection_model.inference("xyz.png")
    print(result)