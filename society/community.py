import inspect
# import gradio as gr
import uuid
import os
import re
import io
import numpy as np
import random
from PIL import Image
import argparse
import openai
import requests


from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI


from society.arxiv_search.agent import ArxivSearch
from society.image_captioning.agent import ImageCaptioning
from society.text_to_image.agent import Text2Image
from society.vqa.agent import mPLUG, OFA, BLIP2