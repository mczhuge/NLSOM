import requests
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import os

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class Whisper:
    def __init__(self, device):
        print(f"Initializing Audio2Text to {device}")
        self.device = "cpu"
        self.API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
        self.headers = {"Authorization": "Bearer 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"} # TODO: Huggingface API


    @prompts(name="Whisper (audio-recognition)",
             description="useful when you want to recognize the context of an audio file. "
                         "Whisper is a general-purpose speech recognition model. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, filename):

        audio_json = self.query(filename)
        # You can access the image with PIL.Image for example

        print(
            f"\nProcessed Audio2Text, Input File: {filename}, Output Content: {audio_json}")
        return audio_json["text"]

    
    def query(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(self.API_URL, headers=self.headers, data=data)
        return response.json()
    

class Paraformer:
    def __init__(self, device):
        print(f"Initializing Paraformer to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online'
        self.pipeline_asr = pipeline('auto-speech-recognition', model=model_id)

    @prompts(name="Paraformer (Chinese audio-recognition)",
             description="useful when you want to recognize the context of an audio file. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, filename):

        result =  self.pipeline_asr(filename)
        return result
    


if __name__ == "__main__":
    asr_model = Paraformer(device="cuda:0")
    result = asr_model.inference("http://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/repo?Revision=master\u0026FilePath=example/asr_example.wav")
    print(result)