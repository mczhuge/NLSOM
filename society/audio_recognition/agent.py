import requests
import os
#from modelscope.pipelines import pipeline

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class Whisper:
    def __init__(self, device="cpu"):
        self.device = device
        self.API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
        self.headers = {"Authorization": "Bearer "+os.getenv("HUGGINGFACE_ACCESS_Tokens")} 


    @prompts(name="Whisper",
             description="useful when you want to recognize the context of an audio file. "
                         "Whisper is a general-purpose speech recognition model. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, filename):

        audio_json = self.query(filename)

        print(
            f"\nProcessed Audio2Text, Input File: {filename}, Output Content: {audio_json}")
        return audio_json["text"]

    
    def query(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(self.API_URL, headers=self.headers, data=data)
        return response.json()
    
    
if __name__ == "__main__":
    #"http://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/repo?Revision=master\u0026FilePath=example/asr_example.wav"
    
    asr_model = Whisper()
    #asr_model = Paraformer(device="cuda:0")
    result = asr_model.inference("sample1.flac")
    print(result)


# class Paraformer:
#     def __init__(self, device="cuda:0"):
#         self.device = device
#         model_id = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online'
#         self.pipeline_asr = pipeline('auto-speech-recognition', model=model_id)

#     @prompts(name="Paraformer",
#              description="useful when you want to recognize the Chinese context of a Chinese audio file. "
#                          "The input to this tool should be a string, representing the image_path. ")
#     def inference(self, filename):

#         result =  self.pipeline_asr(filename)
#         return result