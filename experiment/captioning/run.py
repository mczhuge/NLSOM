import torch
from PIL import Image

import os
import sys
import random
from tqdm import tqdm
import pathlib
import openai
import requests
import time
import numpy as np
from PIL import Image
import datetime
import pandas as pd
import ast
import argparse

sys.path.append(".")
import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor

from utils.prompt_cap import (
                          prompt_from_LLM, 
                          mission_prompt_generation, 
                          mindstorm_prompt_generation, 
                          execution_prompt_generation, 
                          question_prompt_start, _abc_, _123_
                          )

from utils.eval import  evaluate_cap
from utils.data import tara_data


# add your OPENAI_API_KEY
# openai.api_key = os.getenv('OPENAI_API_KEY')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# vlm setting 
_LLM1_ = "InstructGPT"
_VLM1_ = ["BLIP2", "Salesforce/blip2-flan-t5-xl"]
_VLM2_ = ["OFA", 'damo/ofa_visual-question-answering_pretrain_large_en']
_VLM3_ = ["mPLUG", "damo/mplug_visual-question-answering_coco_large_en"]


def main(parser):

    # config
    args = parser.parse_args()
    split = args.split
    dataset_path = args.dataset_path
    cap_txt = args.cap_txt
    mindstorm_turn = args.mindstorm_turn
    group_name = args.group_name
    task = "Image Captioning"

    # load models
    need_VLM = ["Captioning"]

    if group_name in need_VLM:
        
        processor = Blip2Processor.from_pretrained(_VLM1_[1])


        BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(_VLM1_[1], torch_dtype=torch.float16, device_map="auto")

        OFA_MODEL = _VLM2_[1]
        preprocessor = OfaPreprocessor(model_dir=OFA_MODEL)
        ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=OFA_MODEL,
            model_revision='v1.0.1',
            preprocessor=preprocessor)

        mPLUG_MODEL = _VLM3_[1]
        mPLUG_VQA = pipeline('visual-question-answering', model=mPLUG_MODEL)


    # load data
    cap_txt_path = os.path.join(dataset_path, cap_txt)
    with open(cap_txt_path) as f:
        pairs = f.readlines()

    # counting
    count = 0
    right_count = 0


    total_similarity_blip2 = 0 
    total_similarity_mindstorm = 0 

    for item in tqdm(pairs):

        count += 1
        image_path, image_abstract, image_paragraph, ner, place = tara_data(item, split)

        print('===='*10)
        print(str(datetime.datetime.now()).split(".")[0])
        image_file_path = os.path.join(dataset_path, "image/test/"+image_path)

        
        print(image_file_path)

        print('===='*10)
        
        # load sample image
        raw_image = Image.open(image_file_path).convert("RGB")

        if group_name == "Captioning": 

            _LLM_ = "InstructGPT"

            final_goal = "Describe this image in a more informative way, containing high-level reasoning like \"Where is this photo taken?\", \"When is this photo taken?\", \"What's the event or story behind this image?\", etc."

            options = []      

            ########################
            ##    Mission Start   ##
            #########################

            mission_prompt = mission_prompt_generation(task)

            #############################
            ## Task-Oriented mindstorm ##
            #############################

            generated_questions = ["" for _ in range(mindstorm_turn+2)]
            generated_answers = ["" for _ in range(mindstorm_turn+2)]
            mindstorm_record = ["" for _ in range(mindstorm_turn+2)]
            mindstorm_record[0] = mission_prompt
                     
            for j in range(mindstorm_turn):

                question = generated_questions[j] if j!=0 else mission_prompt
                print("\033[48;5;236mInput to the VLMs: \033[1;m", question)
                
                
                # BLIP2 (Agent1)
                input = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
                generated_answer_ids = BLIP2_MODEL.generate(**input, max_new_tokens=20)
                generated_answers[j] = "Answer1: " + processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()
                mindstorm_record[j] +=  " " +  generated_answers[j] + "."
                print("\033[1;33;33m{}: \033[1;m".format(_VLM1_[0]) + generated_answers[j])   
                
                               
                # OFA (Agent2)
                input = {'image': image_file_path, 'text': question}
                generated_answers[j] = generated_answers[j] + "; Answer2: " + ofa_pipe(input)[OutputKeys.TEXT][0]   
                mindstorm_record[j] +=  " " +  "; Answer2: " + ofa_pipe(input)[OutputKeys.TEXT][0] + "."
                print("\033[1;33;33m{}: \033[1;m".format(_VLM2_[0]) + generated_answers[j])
                
                
                # mPLUG (Agent3)
                input = {'image': image_file_path, 'text': question}
                generated_answers[j] = generated_answers[j] + "; Answer3: " + mPLUG_VQA(input)["text"]
                mindstorm_record[j] +=  " " +  "; Answer3: " + mPLUG_VQA(input)["text"] + "."           
                print("\033[1;33;33m{}: \033[1;m".format(_VLM3_[0]) + generated_answers[j])
                
                

                if j == mindstorm_turn-1:
                    generated_questions[j+1] = "Describe this image in a more informative way, including high-level reasoning such as \"Where was this photo taken?\", \"When was this photo taken?\", \"What is the event or story behind this image?\" and any other relevant details."
                    mindstorm_record[j] +=  " " +  generated_questions[j+1]

                else:
                    prologue = generated_answers[0]
                    mindstorm_prompt = mindstorm_prompt_generation(task, final_goal, options, prologue, generated_questions, j, generated_answers)
                    print("\033[48;5;236mInput to the {}: \033[1;m".format(_LLM_), mindstorm_prompt)
                    generated_questions[j+1], model_name = prompt_from_LLM(_LLM_, prompt=mindstorm_prompt)
                    if "?" in generated_questions[j+1]:
                        generated_questions[j+1] = generated_questions[j+1].split("?")[0] + "?"
                    else:
                        generated_questions[j+1] = generated_questions[j+1] + "?"
                    mindstorm_record[j] +=  " " + generated_questions[j+1] 

            ###########################
            ##   Opinion Gathering   ##
            ###########################

            # In the prompt, we use "brainstorm" rather than "Mindstorm" to alleviate LLM's misunderstanding of the purpose.
            analysis_prompt = "There is a brainstorm record: {}".format(" ".join(mindstorm_record)) + "Please analyze and summarize them in a few sentences.".format(question)

            print("\033[48;5;236mInput to the {}: \033[1;m".format(_LLM_), analysis_prompt)

            analysis, model_name = prompt_from_LLM(_LLM_, prompt=analysis_prompt, max_token=100)
            print("\033[1;33;33m{}'s analysis: \033[1;m".format(model_name) + analysis) 


            ############################
            ##     Final Execution    ##
            ############################
            
            print("\033[1;33;33mNer: {}\033[1;m".format(ner))

            options = None
            execution_prompt = execution_prompt_generation(options, analysis, final_goal, task)
            print("\033[48;5;236mInput to the LLM: \033[1;m", str(execution_prompt))
            generated_answers, model_name = prompt_from_LLM(_LLM_, prompt=execution_prompt)
            print("\033[1;33;33m{}: \033[1;m".format(_LLM_) + generated_answers)   


            ###################
            ##      Eval     ##
            ###################

            print("\033[1;33;33mFinal Answer: \033[1;m", str(generated_answers))
            
            # BLIP2 (Agent1)
            input = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
            generated_answer_ids = BLIP2_MODEL.generate(**input, max_new_tokens=30)
            blip2_answer = processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()
            print("\033[1;33;33m{}: \033[1;m".format(_VLM1_[0]) + blip2_answer)   

            blip2_score = evaluate_cap(ner, blip2_answer)
            mindstorm_score = evaluate_cap(ner, generated_answers)

            print("#"*10)
            print("BLIP2: ", blip2_score)
            print("#"*10)
            print("Mindstorm: ", mindstorm_score)


            total_similarity_blip2 = total_similarity_blip2 + blip2_score['captioning']['similarity']
            total_similarity_mindstorm = total_similarity_mindstorm + mindstorm_score['captioning']['similarity']


            current_similarity_blip2 = total_similarity_blip2 / count
            current_similarity_mindstorm = total_similarity_mindstorm / count

            print("\033[1;33;33m{}'s answer: \033[1;m".format(model_name), str(generated_answers))
            print("\033[1;33;33mOriginal answer: {}\033[1;m".format(str(mindstorm_record[0])))
            print("\033[1;33;33mAbstact: {}\033[1;m".format(image_abstract))
            print("\033[1;33;33mParagraph: {}\033[1;m".format(image_paragraph))
                
            print({'Total Count:': count, 
                           'BLIP2-Sim': current_similarity_blip2,  
                           'Mindstorm-Sim': current_similarity_mindstorm,
                           })

        else:
            raise NotImplementedError
        

    print("Total Win Rate: ", 100*(right_count/len(pairs)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset_path", default="data")
    parser.add_argument("--cap_txt", default="tara.txt")

    # experiment_save
    result_path = "./results/"
    timestamp = str(datetime.datetime.now()).split(".")[0]
    random_seed = str(np.random.randint(1,100))

    # setting
    parser.add_argument("--mindstorm_turn", default=10)
    parser.add_argument("--llm_version", default="InstructGPT")
    parser.add_argument("--group_name", default="Captioning")

    main(parser)