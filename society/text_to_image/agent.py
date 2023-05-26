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

class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.API_URL = "https://api-inference.huggingface.co/models/andite/anything-v4.0"
        self.headers = {"Authorization": "Bearer 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"}

        self.a_prompt = 'best quality, extremely detailed'

    @prompts(name="anything-v4.0 (text-to-image)",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        
        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        prompt = str(text + ', ' + self.a_prompt)

        image_bytes = self.query({
	            "inputs": prompt,
            })
        # You can access the image with PIL.Image for example

        image = Image.open(io.BytesIO(image_bytes))
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {prompt}, Output Image: {image_filename}")
        return image_filename

    
    def query(self, payload):
	    response = requests.post(self.API_URL, headers=self.headers, json=payload)
	    return response.content