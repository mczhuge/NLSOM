import uuid
import os
import replicate
import urllib.request
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
  
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator
    

class DeforumSD:
    def __init__(self, device="cpu"):
        self.device = device

    @prompts(name="DeforumSD",
             description="useful when you want to generate a video from a user input text and save it to a file. "
                         "like: generate an video of an object or something, or generate an video that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate video. ")
    def inference(self, text):
        

        script_prompt = "Unlock your creativity with artistic expression, then slighly expand this: " + text
        director = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=script_prompt,
                    temperature=0.5,
                    max_tokens=200,
                    )
        script = director["choices"][0]["text"].strip("\n")

        video_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.mp4")
        
        output = replicate.run(
            "deforum/deforum_stable_diffusion:e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6",
            input={"max_frames": 300,
                   "animation_prompts": str(script)
                   }
        )
        
        urllib.request.urlretrieve(output, video_filename)
        print(output)

        return output#video_filename


if __name__ == "__main__":
    t2v_model = DeforumSD(device="cpu")
    t2v_model.inference("0: "+ "Superman is saving the New York city! ")