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

from utils.prompt_vqa import (
                          prompt_from_LLM, 
                          mission_prompt_generation, 
                          mindstorm_prompt_generation, 
                          execution_prompt_generation, 
                          question_prompt_start, _abc_, _123_
                          )

from utils.eval import true_of_fault
from utils.data import aokvqa_data
from utils.print import print_scores

# add your OPENAI_API_KEY
openai.api_key = os.getenv('OPENAI_API_KEY')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def main(parser):

    # config
    args = parser.parse_args()
    split = args.split
    dataset_path = args.dataset_path
    vqa_txt = args.vqa_txt
    model_save_path = args.model_save_path
    mindstorm_round = args.mindstorm_round
    project_name = args.project_name
    group_name = args.group_name
    
    # vlm setting 
    _LLM1_ = "InstructGPT"
    _VLM1_ = ["BLIP2", "Salesforce/blip2-flan-t5-xl"]
    _VLM2_ = ["OFA", 'damo/ofa_visual-question-answering_pretrain_large_en']
    _VLM3_ = ["mPLUG", "damo/mplug_visual-question-answering_coco_large_en"]

    # load models
    need_LLM = ["VQA"]
    need_VLM = ["VQA"]

    if group_name in need_VLM:
        
        # load BLIP2
        processor = Blip2Processor.from_pretrained(_VLM1_[1])
        BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(_VLM1_[1], torch_dtype=torch.float16, device_map="auto")
        
        # load OFA
        OFA_MODEL = _VLM2_[1]
        preprocessor = OfaPreprocessor(model_dir=OFA_MODEL)
        ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=OFA_MODEL,
            model_revision='v1.0.1',
            preprocessor=preprocessor)
        
        # load mPLUG
        mPLUG_MODEL = _VLM3_[1]
        mPLUG_VQA = pipeline('visual-question-answering', model=mPLUG_MODEL)


    # load data
    vqa_txt_path = dataset_path+vqa_txt
    with open(vqa_txt_path) as f:
        pairs = f.readlines()

    # counting
    count = 0
    right_count = 0
    failure_count = 0

    # failure_cases
    wrong_dict = []
    problem_dict = dict() 

    for item in tqdm(pairs):

        count += 1
        image_path, VQA_question, answer_options, answer_idx = aokvqa_data(item, split)

        print('===='*10)
        print(str(datetime.datetime.now()).split(".")[0])
        image_file_path = os.path.join(dataset_path, image_path)

        VQA_or_QA = "QA"
        print(image_file_path)
        if image_file_path.split("/")[-1] != "no_image.jpeg":
            VQA_or_QA = "VQA"
        print("\033[1;34;36m{} question: \033[1;m".format(VQA_or_QA), VQA_question)                     
        print('===='*10)
        
        # load sample image
        raw_image = Image.open(image_file_path).convert("RGB")


        if group_name == "VQA": 

            _LLM_ = "InstructGPT"
            _VLM1_ = "BLIP2"
            _VLM2_ = "OFA"
            _VLM3_ = "mPLUG"

            options = []      
           
            print(answer_options)

            for oid in range(10):
                if oid < len(answer_options):
                    options.append(_abc_[str(oid)] + answer_options[oid])
                else:
                    answer_options.append("_NONE_")

            print(answer_options)
            print("\033[1;34;36mOptions: \033[1;m" + '{' + str(options).replace("[", "").replace("]", "") + '}')

            ###############################
            ##         VQA Start         ##
            ###############################

            mission_prompt = mission_prompt_generation(VQA_or_QA)

            ###############################
            ## Task-Oriented Mindstorm  ##
            ###############################

            generated_questions = ["" for _ in range(mindstorm_round+2)]
            generated_answers = ["" for _ in range(mindstorm_round+2)]
            mindstorm_record = ["" for _ in range(mindstorm_round+2)]
            mindstorm_record[0] = mission_prompt
                  
            for j in range(mindstorm_round):

                question = generated_questions[j] if j!=0 else mission_prompt
                print("\033[48;5;236mInput to the {}: \033[1;m".format(_VLM3_), question)
                
                
                # BLIP2 (Agent1)
                input = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
                generated_answer_ids = BLIP2_MODEL.generate(**input, max_new_tokens=20)
                generated_answers[j] = " Answer1: " + processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()
                mindstorm_record[j] +=  " " +  generated_answers[j]
                print("\033[1;33;33m{}: \033[1;m".format(_VLM1_) + generated_answers[j])   
                

                
                # OFA (Agent2)
                input = {'image': image_file_path, 'text': question}
                generated_answers[j] = generated_answers[j] + "; Answer2: " + ofa_pipe(input)[OutputKeys.TEXT][0]   
                mindstorm_record[j] +=   "; Answer2: " + ofa_pipe(input)[OutputKeys.TEXT][0]
                print("\033[1;33;33m{}: \033[1;m".format(_VLM2_) + generated_answers[j])
                
                
                # mPLUG (Agent3)
                input = {'image': image_file_path, 'text': question}
                generated_answers[j] = generated_answers[j] + "; Answer3: " + mPLUG_VQA(input)["text"]
                mindstorm_record[j] += "; Answer3: " + mPLUG_VQA(input)["text"] + "."         
                print("\033[1;33;33m{}: \033[1;m".format(_VLM3_) + generated_answers[j])
                  


                if j == mindstorm_round-1:
                    generated_questions[j+1] = VQA_question
                    mindstorm_record[j] +=  " " +  generated_questions[j+1]

                else:
                    prolugue = generated_answers[0]
                    mindstorm_prompt = mindstorm_prompt_generation(VQA_or_QA, VQA_question, options, prolugue, generated_questions, j, generated_answers)
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
            analysis_prompt = "There is a brainstorm record: {}".format(" ".join(mindstorm_record)) + "Please summarize them in a few sentences.".format(question)

            print("\033[48;5;236mInput to the {}: \033[1;m".format(_LLM_), analysis_prompt)

            analysis, model_name = prompt_from_LLM(_LLM_, prompt=analysis_prompt, max_token=100)
            print("\033[1;33;33m{}'s analysis: \033[1;m".format(model_name) + analysis) 

            ###################
            ##   Execution   ##
            ###################
            
            execution_prompt = execution_prompt_generation(options, analysis, VQA_question, VQA_or_QA)


            print("\033[48;5;236mInput to the LLM: \033[1;m", str(execution_prompt))
            generated_answers, model_name = prompt_from_LLM(_LLM_, prompt=execution_prompt)
            print("\033[1;33;33m{}'s answer: \033[1;m".format(model_name), str(generated_answers))
        
            ###################
            ##      Eval     ##
            ###################

            print("\033[1;33;33mOptions: \033[1;m", str(options))
            print("\033[1;33;33mFinal Answer: \033[1;m", str(generated_answers))
            generated_answers, problem_dict, failure_count = true_of_fault(item, answer_options, generated_answers, problem_dict, dataset_path, failure_count, options, answer_idx)

            right_count += 1 if generated_answers==answer_options[int(answer_idx)] else 0

       
            # Show other VLM's answer
            question = "Question: " + VQA_question + "Options: " + str(options).replace("[", "").replace("]", "") + " Context: Select the best answer. " + "Answer: "

            # BLIP2 (Agent1)
            input = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
            generated_answer_ids = BLIP2_MODEL.generate(**input, max_new_tokens=20)
            generated_answers_blip2 = processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0].strip()
            print("\033[1;33;33m{}: \033[1;m".format(_VLM1_) + generated_answers_blip2)   
            
                            
            # OFA (Agent2)
            input = {'image': image_file_path, 'text': question}
            generated_answers_ofa =  ofa_pipe(input)[OutputKeys.TEXT][0]   
            print("\033[1;33;33m{}: \033[1;m".format(_VLM2_) + generated_answers_ofa)
            
            
            # mPLUG (Agent3)
            input = {'image': image_file_path, 'text': question}
            generated_answers_mplug = mPLUG_VQA(input)["text"]
            print("\033[1;33;33m{}: \033[1;m".format(_VLM3_) + generated_answers_mplug)


            current_wr = (right_count/count) * 100
            current_fc = (failure_count/count) * 100
            print(right_count, count)
            print(f"Current Win Rate: {current_wr:.5f}%, Current Failure Rate: {current_fc:.5f}%")

        else:
            raise NotImplementedError
        

    print("Total Win Rate: ", 100*(right_count/len(pairs)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset_path", default="/home/zhugem/L2T/MM-SOM/data")
    parser.add_argument("--vqa_txt", default="/aokvqa/val.txt")

    # experiment_save
    result_path = "./results/"
    timestamp = str(datetime.datetime.now()).split(".")[0]
    random_seed = str(np.random.randint(1,100))
    model_save_path = os.path.join(result_path, timestamp, random_seed)
    parser.add_argument("--model_save_path",  default=model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    # setting
    parser.add_argument("--mindstorm_round", default=5)
    parser.add_argument("--mode", default="QCM")
    parser.add_argument("--project_name", default="NLSOM")
    parser.add_argument("--group_name", default="VQA")

    main(parser)
