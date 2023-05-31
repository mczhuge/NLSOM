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

class AnythingV4:
    def __init__(self, device="cpu"):
        self.device = device
        self.API_URL = "https://api-inference.huggingface.co/models/andite/anything-v4.0"
        self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

        self.a_prompt = 'best quality, extremely detailed'

    @prompts(name="AnythingV4",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        
        image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
        #image_filename = "result.png"
        prompt = str(text + ', ' + self.a_prompt)

        image_bytes = self.query({
	            "inputs": prompt,
            })

        image = Image.open(io.BytesIO(image_bytes))
        image.save(image_filename)
        return image_filename

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.content
    
    
# class AnythingV3:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/Linaqruf/anything-v3.0"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="AnythingV3",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content
    
    
# class OpenJourneyV4:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney-v4"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="OpenJourneyV4",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content
    
    
# class OpenJourney:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="OpenJourney",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content
    
    
# class StableDiffusionV15:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="StableDiffusionV15",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content


# class StableDiffusionV21B:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="StableDiffusionV21B",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content


# class StableDiffusionV21:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
#         self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")}

#         self.a_prompt = 'best quality, extremely detailed'

#     @prompts(name="StableDiffusionV21",
#              description="useful when you want to generate an image from a user input text and save it to a file. "
#                          "like: generate an image of an object or something, or generate an image that includes some objects. "
#                          "The input to this tool should be a string, representing the text used to generate image. ")
#     def inference(self, text):
        
#         image_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")
#         #image_filename = "result.png"
#         prompt = str(text + ', ' + self.a_prompt)

#         image_bytes = self.query({
# 	            "inputs": prompt,
#             })

#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(image_filename)
#         return image_filename

#     def query(self, payload):
#         response = requests.post(self.API_URL, headers=self.headers, json=payload)
#         return response.content


if __name__ == "__main__":
     
     t2i = AnythingV4()
     t2i.inference("Chinese SuperMan in the Old Street.")