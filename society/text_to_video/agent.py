import shutil
import uuid
import os
import io
import torch
import replicate
import urllib.request


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class DeforumSD:
    def __init__(self, device):
        print(f"Initializing DeforumSD to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32

    @prompts(name="DeforumSD (text-to-video)",
             description="useful when you want to generate a video from a user input text and save it to a file. "
                         "like: generate an video of an object or something, or generate an video that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate video. ")
    def inference(self, text):
        
        video_filename = os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")#f"{str(uuid.uuid4())[:8]}.mp4" #os.path.join('data', f"{str(uuid.uuid4())[:8]}.png")

        output = replicate.run(
            "deforum/deforum_stable_diffusion:e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6",
            input={"max_frames": 100,
                   "animation_prompts": str(text)
                   }
        )

        urllib.request.urlretrieve(output,  video_filename)
        print(output)

        return output#video_filename


if __name__ == "__main__":
    t2v_model = DeforumSD(device="cuda:0")
    t2v_model.inference("0: "+ "Superman is saving the New York city! ")
    #video_filename = f"{str(uuid.uuid4())[:8]}.mp4"
    #output = "https://replicate.delivery/pbxt/nX9Ast09y1bkApFMxv0i9zWOk6DRvW1wI6wrwPHpBJsf4mfQA/out.mp4"
    #urllib.request.urlretrieve(output,  video_filename)