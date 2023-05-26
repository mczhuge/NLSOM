from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
import torch
from PIL import Image

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class BLIP2_VQA:
    def __init__(self, device):
        print(f"Initializing BLIP2 to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'Salesforce/blip2-flan-t5-xl'
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    @prompts(name="BLIP2_VQA",
             description="Useful when you want to ask some question about the image. Receives image_path and language-based question as inputs. "
                         "When using this model, you can also consider to ask more question according to previous question."
                         "If there are other VQA methods, it is also suggested to utilize other VQA methods to enhance your ability to answer the questions."
                         "The input to this tool should be a string, representing the image_path.")
    def inference(self, input):
        try:
            image_path, question = input.split(",")[0].strip(), input.split(",")[1].strip()
            image_path = image_path.strip("\n")
            raw_image = Image.open(image_path).convert("RGB")
        except:
            print("No question as input, use the template: \"Describe this image in details\" as question")
            image_path = input.strip().strip("\n")
            raw_image = Image.open(image_path).convert("RGB")
            question = "Describe this image in details."
            

        input = self.processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
        generated_answer_ids = self.BLIP2_MODEL.generate(**input, max_new_tokens=20)
        answer = self.processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()

        return answer



class mPLUG_VQA:
    def __init__(self, device):
        print(f"Initializing mPLUG to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/mplug_visual-question-answering_coco_large_en'
        self.pipeline_caption = pipeline(Tasks.visual_question_answering, model=model_id)

    @prompts(name="mPLUG_VQA",
             description="Useful when you want to ask some question about the image. Receives image_path and language-based question as inputs. "
                         "When using this model, you can also consider to ask more question according to previous question."
                         "If there are other VQA methods, it is also suggested to utilize other VQA methods to enhance your ability to answer the questions."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, input):
        try:
            image_path, question = input.split(",")[0].strip(), input.split(",")[1].strip()
        except:
            print("No question as input, use the template: \"Describe this image in details\" as question")
            image_path = input.strip()
            question = "Describe this image in details."
        #else:
            
        image_path = image_path.strip("\n")
        input = {'image': image_path, 'text': question}
        answer = self.pipeline_caption(input)["text"]
        print(f"\nProcessed VQA, Input Image: {image_path}, Input Question: {question}, Output Text: {answer}")
        return answer


class OFA_VQA:
    def __init__(self, device):
        print(f"Initializing OFA to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        model_id = 'damo/ofa_visual-question-answering_pretrain_large_en'
        self.preprocessor = OfaPreprocessor(model_dir=model_id)
        self.ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=model_id,
            model_revision='v1.0.1',
            preprocessor=self.preprocessor)

    @prompts(name="OFA_VQA",
             description="Useful when you want to ask some question about the image. Receives image_path and language-based question as inputs. "
                         "When using this model, you can also consider to ask more question according to previous question."
                         "If there are other VQA methods, it is also suggested to utilize other VQA methods to enhance your ability to answer the questions."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, input):
        try:
            image_path, question = input.split(",")[0].strip(), input.split(",")[1].strip()
        except:
            print("No question as input, use the template: \"Describe this image in details\" as question")
            image_path = input.strip()
            question = "Describe this image in details."
            
        input = {'image': image_path, 'text': question}
        answer = self.ofa_pipe(input)[OutputKeys.TEXT][0]  
        print(f"\nProcessed VQA, Input Image: {image_path}, Input Question: {question}, Output Text: {answer}")
        return answer


