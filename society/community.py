"""
In these natural language-based societies of mind (NLSOMs), new agents—--all communicating through the same universal symbolic language—are easily added in a modular fashion.  
We view this as a starting point towards much larger NLSOMs with billions of agents—some of which may be humans.
"""

import inspect
#import gradio as gr
import uuid
import os
import re
import io
import numpy as np
import random
import torch
from PIL import Image
import argparse
#import openai
import requests
#import easyocr
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



# 17 communities / 32 agents
from society.audio_recognition.agent import Whisper, Paraformer # 2
from society.body_reshaping.agent import SAFG # 1
from society.image_captioning.agent import mPLUG_captioning, OFA_distilled_captioning, OFA_large_captioning, BLIP2_captioning # 4
from society.image_colorization.agent import DDColor # 1
from society.image_deblur.agent import NAFNet # 1
from society.image_to_3d.agent import HumanReconstruction # 1
from society.object_detection.agent import DETR # 1
from society.ocr.agent import EasyOCR # 1
from society.role_play.agent import LiuBei, GuanYu, ZhangFei, ZhugeLiang #4
from society.search.agent import SE_A, SE_B, SE_C, SE_D #4
from society.sentence_refine.agent import SentenceRefine # 1
from society.skin_retouching.agent import ABPN # 1
from society.text_to_image.agent import Text2Image # 1
from society.vqa.agent import OFA_VQA, BLIP2_VQA, mPLUG_VQA # 3
from society.text_to_speech.agent import Text2Speech #1
from society.text_to_video.agent import DeforumSD #1
from society.vqa.agent import BLIP2_VQA, mPLUG_VQA, OFA_VQA #3



