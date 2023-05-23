import requests

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class Audio2Text:
    def __init__(self, device):
        print(f"Initializing Audio2Text to {device}")
        self.device = "cpu"
        self.API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
        self.headers = {"Authorization": "Bearer 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"} # TODO: Huggingface API


    @prompts(name="Whisper (audio-to-text)",
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