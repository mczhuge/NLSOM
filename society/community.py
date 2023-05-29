"""
In these natural language-based societies of mind (NLSOMs), new agents—--all communicating through the same universal symbolic language—are easily added in a modular fashion.  
We view this as a starting point towards much larger NLSOMs with billions of agents—some of which may be humans.
"""

# 16 communities / 34 agents

# CPU Only
from society.audio_recognition.agent import Whisper # 1
from society.object_detection.agent import DETR # 1
from society.ocr.agent import EasyOCR # 1
from society.role_play.agent import LiuBei, GuanYu, ZhangFei, ZhugeLiang #4
from society.search.agent import SE_A, SE_B, SE_C, SE_D #4
from society.sentence_refine.agent import SentenceRefine # 1
from society.text_to_image.agent import AnythingV4 #1 & [6 Candidates] AnythingV3, OpenJourneyV4, OpenJourney, StableDiffusionV15, StableDiffusionV21B, StableDiffusionV21 # 7
from society.text_to_speech.agent import TTS #1
from society.text_to_video.agent import DeforumSD #1
 
# Need GPU
from society.body_reshaping.agent import SAFG # 1
from society.image_captioning.agent import OFA_large_captioning  # 1 & [3 Candiates] OFA_distilled_captioning, BLIP2_captioning, mPLUG_captioning, OFA_large_captioning
from society.image_colorization.agent import DDColor # 1 & [1 Candiates] UNet
from society.image_deblur.agent import NAFNet # 1
from society.image_to_3d.agent import HRNet # 1
from society.skin_retouching.agent import ABPN # 1
from society.vqa.agent import BLIP2_VQA, mPLUG_VQA, OFA_VQA #3



