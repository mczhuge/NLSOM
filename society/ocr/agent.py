import requests
import torch
import os
import easyocr

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class EasyOCR:
    def __init__(self, device="cpu"):
        self.ocr = easyocr.Reader(['ch_sim','en']) 

    @prompts(name="EasyOCR (OCR)",
             description="useful when you want to recognize the word or text in an image. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, filename):

        text = ""
        result = self.ocr.readtext(filename)
        for item in result:
            text += " " + item[1]
        return text.strip()



if __name__ == "__main__":
    ocr_model = EasyOCR()
    result = ocr_model.inference('00155719.png')
    print(result)