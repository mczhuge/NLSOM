# https://github.com/coqui-ai/TTS
import uuid
import os
import io
from TTS.api import TTS
# Running a multi-speaker and multi-lingual model


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class TTS:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = "cpu"
        model_name = TTS.list_models()[0]
        self.tts = TTS(model_name)


    @prompts(name="TTS",
             description="useful when you want to generate an audio from a user input text and save it to a file. "
                         "The input to this tool should be a string, representing the text used to generate audio. ")
    def inference(self, text):
        
        audio_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.wav")
        #audio_filename = f"{str(uuid.uuid4())[:8]}.wav"

        self.tts.tts_to_file(text=text, speaker=self.tts.speakers[0], language=self.tts.languages[0], file_path=audio_filename)

        return audio_filename

if __name__ == "__main__":
    tts_model = TTS(device="cuda:0")
    image = tts_model.inference("Wo zong shi lin shi bao fo jiao, lin shi bao fo jiao.")