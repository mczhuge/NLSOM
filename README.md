# <p align=center>`Mindstorms in Natural Language-based Societies of Mind`</p><!-- omit in toc -->
![overview](config/som.svg)
> What magical trick makes us intelligent?  The trick is that there is no trick.  The power of intelligence stems from our vast diversity, not from any single, perfect principle. — Marvin Minsky, The Society of Mind, p. 308

![](https://i.imgur.com/waxVImv.png)
[![KAUST-AINT](https://cemse.kaust.edu.sa/themes/custom/bootstrap_cemse/logo.svg)](https://cemse.kaust.edu.sa/ai)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2304.12995)
<a href="https://github.com/reworkd/AgentGPT/blob/master/docs/README.zh-HANS.md"><img src="https://img.shields.io/badge/lang-简体中文-red.svg" alt="简体中文"></a>


### ✨ Introduction

This project is the **technical extenstion** for original [NLSOM paper](): allowing you to build a specific NLSOM quickly. 
When you provide *a file* and *a target*, LLM automates all these processes for you:

- **🧰 Recommendation**: Autonomously select communities and agents to form a self-organized NLSOM for solving the target.
- **🧠 Mindstorm**: Empower with the automated mindstorm. Multiple agents (models or APIs) can effortlessly collaborate to solve tasks together.
- **💰 Reward**: Rewards are given to all agents involved.

Features:
- [x] Rule your NLSOM: effortlessly organize an NLSOM in various fields by simply changing the template.
- [x] Easy to extend: customise your own community and agents.
- [x] Reward Design: provide a reward mechanism (albeit rough). You can easily upgrade to a more refined version.
- [x] An elegant UI: facilitate better visualization and support for diverse media sources (image, text, audio, video, docs, pdf, etc).


### 💡️ Important Concepts

- We introduce the concepts of *society, community and agent*.
- *Agents* in the same *community* will collaborate 



### 💾 Usage

#### 1. Install
```
conda create -n nlsom python=3.8
pip install colorlog==6.7.0
pip install langchain==0.0.158
pip install sqlalchemy==2.0.12
pip install openai
pip install guidance
pip install wolframalpha
pip install wikipedia
pip install bs4
pip install streamlit==1.22.0
pip install streamlit_chat==0.0.2.2
pip install colorama
pip install torch==1.13.1
pip install torchvision==0.14.1
pip install transformers
pip install modelscope[cv] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope[nlp] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install easyocr
pip install tinydb
pip install deeplake
pip install python-dotenv
pip install watchdog
pip install unstructured
pip install pdf2image==1.16.3
pip install pytesseract==0.3.10
pip install tabulate
pip install tesseract


langchain
pip3 install azure-storage-blob azure-identity
pip install azure-ai-formrecognizer==3.2.0
pip install  azure_ai_vision
pip install azure-cognitiveservices-vision-customvision
pip install azure-cognitiveservices-speech
pip3 install azure-ai-textanalytics==5.2.0b2
pip install ipython
#text-to-protein
git clone git@github.com:dptech-corp/Uni-Core.git
pip install biopython
#text-to-video
pip install ipdb
pip install open_clip_torch
pip install pytorch-lightning
#ocr
pip install easyocr
#ofa-ocr
pip install unicodedata2
pip install zhconv
pip install decord>=0.6.0
#cv_fft_inpainting_lama
pip install kornia
#damo/cv_mdm_motion-generation
pip install smplx
pip install git+https://github.com/openai/CLIP.git
pip install chumpy
pip install ffmpeg
#cv_hrnet_image-human-reconstruction
pip install trimesh
pip3 install pymcubes
#wikipedia api
pip install wikipedia
#wolframalpha
pip install wolframalpha
#!pip install arxiv
https://github.com/microsoft/TaskMatrix/issues/179
#Auto-GPT
pip install langchain==0.0.158
sqlalchemy 2.0.12
#tinydb
pip install tinydb
```

```
streamlit run app.py
```


#### 2-1. Focus more on NLSOM

改下载路径
```
>>> import modelscope
>>> print(modelscope.__file__)
/home/zhugem/anaconda3/envs/automind/lib/python3.8/site-packages/modelscope/utils/file_utils.py
default_cache_dir = Path.home().joinpath('/ibex/ai/home/zhugem/AutoMind/checkpoints', 'modelscope')
```

```
import transformers
print(transformers.__file__)
/home/zhugem/anaconda3/envs/automind/lib/python3.8/site-packages/transformers/__init__.py
/home/zhugem/anaconda3/envs/automind/lib/python3.8/site-packages/transformers/utils/hub.py
torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "/ibex/ai/home/zhugem/AutoMind/checkpoints"), "torch"))
hf_cache_home = os.path.expanduser(
   os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "/ibex/ai/home/zhugem/AutoMind/checkpoints"), "huggingface"))
)

```

```
import langchain
print(langchain.__file__)
cp -r automind/* /home/zhugem/anaconda3/envs/automind/lib/python3.8/site-packages/langchain/
```

#### 2-2. Focus more on Mindstorm

##### 1. Model Collaboration

##### 2. API Collaboration

##### 3. Role-Play Collaboration




### 
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa
```

```
wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz
tar -zxvf git-lfs-linux-amd64-v3.3.0.tar.gz
```
```
https://github.com/git-lfs/git-lfs/releases
cd git-lfs-3.3.0 
vim install.sh
-> prefix="/ibex/ai/home/zhugem/"
vim ~/.bashrc
-> PATH+=:"/ibex/ai/home/zhugem/bin"
source ~/.bashrc
```

```
srun -p batch -t 48:00:00 --gres=gpu:1 --reservation=A100 --cpus-per-gpu=12 --mem=128G --pty bash -l
```

```
srun -p batch -t 2:00:00 --gres=gpu:1 --constraint="v100" --cpus-per-task 4 --mem=24G --pty bash -l
```


```
cd /ibex/ai/home/zhugem
```

Sometines, the ssh gets unconnected.
```
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa
```

## Contribute
Please feel free to submit a pull request if you can optimize the identified issue. We are eager to promptly incorporate any improvements. 
* Add more communities and agents
* Optimize the prompt of the Mindstorm
* Design a more accurate reward mechanism
* Make the NLSOM learnable



## Preliminary Experiments in Paper 

## :black_nib: Citation

Reference to cite:

```

```

## Acknowledgments

This project utilizes parts of code from the following open-source repositories:

[langchain](https://github.com/hwchase17/langchain), [BabyAGI](https://github.com/yoheinakajima/babyagi), [TaskMatrix](https://github.com/microsoft/TaskMatrix), [DataChad](https://github.com/gustavz/DataChad), [streamlit](https://github.com/streamlit/streamlit).

We also thank great AI platforms:

[huggingface](https://github.com/huggingface/transformers), [modelscope](https://github.com/modelscope/modelscope).

