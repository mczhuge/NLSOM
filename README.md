# <p align=center>`Mindstorms in Natural Language-based Societies of Mind`</p><!-- omit in toc -->
![overview](config/nlsom.svg)
> What magical trick makes us intelligent?  The trick is that there is no trick.  The power of intelligence stems from our vast diversity, not from any single, perfect principle. — Marvin Minsky, The Society of Mind, p. 308

![](https://i.imgur.com/waxVImv.png)
[![KAUST-AINT](https://cemse.kaust.edu.sa/themes/custom/bootstrap_cemse/logo.svg)](https://cemse.kaust.edu.sa/ai)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2304.12995)
<a href="https://github.com/reworkd/AgentGPT/blob/master/docs/README.zh-HANS.md"><img src="https://img.shields.io/badge/lang-简体中文-red.svg" alt="简体中文"></a>

We are the multimodal team at the KAUST AI Initiative, led by Prof. [Jürgen Schmidhuber](https://scholar.google.com/citations?user=gLnCTgIAAAAJ&hl=en&oi=ao).


## ✨ Introduction

This project is the **technical extenstion** for original [NLSOM paper](): allowing you to build a specific NLSOM quickly. 
When you provide the inputs (files or targets), LLM automates all these processes for you:

- **🧰 Recommendation**: Autonomously select communities and agents to form a self-organized NLSOM for solving the target.
- **🧠 Mindstorm**: Empower with the automated mindstorm. Multiple agents (models or APIs) can effortlessly collaborate to solve tasks together.
- **💰 Reward**: Rewards are given to all agents involved.

Features:
- [x] Rule your NLSOM: effortlessly organize an NLSOM in various fields by simply changing the template.
- [x] Easy to extend: customise your own community and agents (now we have 17 communities and 32 agents). 
- [x] Reward Design: provide a reward mechanism (albeit rough). You can easily upgrade to a more refined version.
- [x] An elegant UI: facilitate better visualization and support for diverse media sources (image, text, audio, video, docs, pdf, etc).


## 💡️ Important Concepts

- We introduce the concepts NLSOM, which contains society, community and agent.
- Agents will collaborate to solve the task, we called it Mindstorm. 
- Jürgen also proposed the Economy of minds (EOM, sec 3 in paper), but we have yet to implement it.



## 🤖 Demo
### 1. Focus more on NLSOM

<details>
    <summary>Demo 1: Society of Mind</summary>
    <p>
        <ul>
            <li>2022.4.28: Add support of inference on **Hugging Face transformers**. For how to use it, please refer to the doc [transformers.md](transformers.md) and our [Hugging Face models](https://huggingface.co/OFA-Sys).</li>
        </ul>
    </p>
</details>

### 2. Focus more on Mindstorm

<details>
    <summary>Demo 2: Model Collaboration</summary>
    <p>
        <ul>
            <li>2022.4.28: Add support of inference on **Hugging Face transformers**. For how to use it, please refer to the doc [transformers.md](transformers.md) and our [Hugging Face models](https://huggingface.co/OFA-Sys).</li>
        </ul>
    </p>
</details>


<details>
    <summary><b>
Demo 3: Collaborative API Usages (Introduce "AGI")</b></summary>
    <img src="https://media.discordapp.net/attachments/1090896867753213973/1111773821859541012/image.png?width=1620&height=436" alt="some_text">
    <p>
        <ul>
            <li><b>🔴 User:</b> Introduce AGI, its definition, and recent work. </li>
        </ul>
    </p>
    <p>
        <ul>
            <li><b>🧰 NLSOM System (Recommendation):</b> Based on this objective, I recommend that NLSOM includes the following AI communities: <u>(1) search</u></li>
        </ul>
    </p>   
    <p>
        <ul>
            <li><b>⚪️ NLSOM System (Self-Organization):</b> We load the recommended AI communities with their their corresponding agents: (a) Arxiv, (b) Wikipedia, (c) WolframAlpha, (d) BingSearch
    </li>
        </ul>
    </p>     
    <p>
        <ul>
            <li><b>🟢 Wikipedia:</b> ['Page: Artificial general intelligence', 'Summary: An artificial general intelligence (AGI) is a type of hypothetical intelligent agent. The AGI concept is that it can learn to accomplish any intellectual task that human beings or other animals can perform. Alternatively, AGI has been defined as an autonomous system that surpasses human capabilities in the majority of economically valuable tasks. Creating AGI is a primary goal of some artificial intelligence research and companies such as OpenAI, DeepMind, and Anthropic. AGI is a common topic in science fiction and futures studies.', 'The timeline for AGI development remains a subject of ongoing debate among researchers and experts. Some argue that it may be possible in years or decades, others maintain it might take a century or longer, and a minority believe it may never be achieved. Additionally, there is debate regarding whether modern deep learning systems, such as GPT-4, are an early yet incomplete form of AGI or if new approaches are required.Contention exists over the potential for AGI to pose a threat to humanity; for example, OpenAI treats it as an existential risk, while others find the development of AGI to be too remote to present a risk.A 2020 survey identified 72 active AGI R&D projects spread across 37 countries.', '', 'Page: Generative artificial intelligence']
    </li>
        </ul>
    </p>  
    <p>
        <ul>
            <li><b>🟡 BingSeach:</b> An artificial general intelligence (AGI) is<b> a type of hypothetical intelligent agent.</b> The AGI concept is that it can learn to accomplish any intellectual task that human beings or other animals can perform. الذكاء العام المصطنع (AGI) (بالإنجليزية: <b>Artificial general intelligence</b> )‏ هو ذكاء الآلة التي لديها القدرة على فهم أو تعلم أي مهمة فكرية يمكن للإنسان القيام بها. Strong artificial intelligence (AI), also known as artificial general intelligence (AGI) or general AI, is<b> a theoretical form of AI used to describe a certain mindset of AI development.</b> If researchers are able to develop Strong AI, the machine would require an intelligence equal to humans; it would have a self-aware consciousness that has the ... <b>Artificial general intelligence</b>, or AGI, is closer than we think, AI experts told Insider. The definition of the hot term AGI is still murky. But experts agree that AGI presents dangers to ... <b>Artificial general intelligence</b> (AGI), or <b>general</b> AI, is a theoretical form of AI where a machine would have an <b>intelligence</b> equaled to humans; it would have a self-aware consciousness that has the ability to solve problems, learn, and plan for the future. An <b>Artificial General Intelligence</b> (AGI) would be a machine capable of understanding the world as well as any human, and with the same capacity to learn how to carry out a huge range of tasks. What exactly is Artificial General Intelligence (AGI)? Artificial General Intelligence is<b> on-par with human capabilities, i.e.</b> AGI<b> system can perform any task that a human can.</b> However, AGI systems can perform the task with greater efficacy than humans only for a particular/single function assigned to them, while they have zero capability to do ... <b>Artificial general intelligence</b> is one of the types of AI that will contribute to the eventual development of <b>artificial</b> superintelligence. In 2022, this vision came much closer to reality, fueled by developments in generative AI that took the world by storm. Our mission is to ensure that <b>artificial general intelligence</b>—AI systems that are generally smarter than humans—benefits all of humanity. If AGI is successfully created, this technology could help us elevate humanity by increasing abundance, turbocharging the global economy, and aiding in the discovery of new scientific knowledge that ...
    </li>
        </ul>
    </p>   
    <p>
        <ul>
            <li><b>🔵 WolframAlpha:</b>  
    </li>
        </ul>
    </p> 
    <p>
        <ul>
            <li><b>🟤 Arxiv:</b>  
    </li>
        </ul>
    </p> 
    <p>
        <ul>
            <li><b>⚪️ NLSOM System (Review):</b>  The NLSOM has successfully solved the user-defined objective by employing all the agents from [Arxiv, Wikipedia, WolframAlpha, BingSearch]. The Arxiv agent provided a paper related to AGI, the Wikipedia agent provided a detailed description of AGI, the WolframAlpha agent provided a definition of AGI, and the BingSearch agent provided a comprehensive overview of AGI.
    </li>
        </ul>
    </p>  
     <p>
        <ul>
            <li><b>⚪️ NLSOM System (Summary):</b>  Artificial General Intelligence (AGI) is a type of hypothetical intelligent agent that can learn to accomplish any intellectual task that human beings or other animals can perform. It is a primary goal of some artificial intelligence research and companies such as OpenAI, DeepMind, and Anthropic. AGI is a common topic in science fiction and futures studies. It is on-par with human capabilities, i.e. AGI system can perform any task that a human can.
    </li>
        </ul>
    </p> 
     <p>
        <ul>
            <li><b>💰 NLSOM System (Reward):</b>  ["Arxiv": 2, "Wikipedia": 3, "WolframAlpha": 0, "BingSearch": 3]
    </li>
        </ul>
    </p> 
</details>




<details>
    <summary><b>
Demo 4: Collaborative Role-Play (The Three Kingdoms)</b></summary>
    <img src="https://media.discordapp.net/attachments/1090896867753213973/1111773821859541012/image.png?width=1620&height=436" alt="some_text">
    <p>
        <ul>
            <li><b>🔴 User:</b> If you find yourself in the Three Kingdoms period, what would be the most effective approach to defeat Cao Cao?</li>
        </ul>
    </p>
    <p>
        <ul>
            <li><b>🧰 NLSOM System (Recommendation):</b> Based on this objective, I recommend that NLSOM includes the following AI communities: <u>(1) Role-play</u></li>
        </ul>
    </p>   
    <p>
        <ul>
            <li><b>⚪️ NLSOM System (Self-Organization):</b> We load the recommended AI communities with their their corresponding agents: (a) GuanYu, (b) LiuBei, (c) ZhugeLiang, (d) ZhangFei
    </li>
        </ul>
    </p>     
    <p>
        <ul>
            <li><b>🟢 GuanYu:</b> The best way to defeat Cao Cao is to use both military and diplomatic tactics. I have witnessed first hand the power of Liu Bei's forces and I believe that a combination of these strengths will be the most effective way to defeat Cao Cao. Through strategic alliances and careful coordination within our own forces, we can utilize our strengths to overcome the enemy and ensure victory. Furthermore, I believe that the element of surprise is also essential, as it allows us to catch the enemy off guard and capitalize on any weaknesses they may have. With the right strategies and strategies in place, I am confident that we can achieve victory against Cao Cao.
    </li>
        </ul>
    </p>  
    <p>
        <ul>
            <li><b>🟡 LiuBei:</b> If I were to find myself in the Three Kingdoms period, the most effective approach to defeat Cao Cao would be to combine the strength of my loyal officers, the wisdom of my advisers, and the courage of my people. I believe in the power of unity and dedication, which will bring us together as one and give us the strength to defeat any obstacle that stands in our way. With this approach, I believe that we can defeat Cao Cao and bring justice to our people.
    </li>
        </ul>
    </p>   
    <p>
        <ul>
            <li><b>🔵 ZhugeLiang:</b> In order to defeat Cao Cao, I would rely on the same strategies that I used to great success during the Three Kingdoms period. I would use a combination of tactical warfare, psychological warfare, and political maneuvering. By making use of my intelligence and strategic acumen, I could outmaneuver and outsmart Cao Cao, slowly whittling away at his forces until I had achieved an ultimate victory. My intelligence is unparalleled, and I am confident that I could successfully defeat Cao Cao and any other enemies that may arise.
    </li>
        </ul>
    </p> 
    <p>
        <ul>
            <li><b>🟤 ZhangFei:</b>  In order to defeat Cao Cao, I would use a combination of strategy and guile. My strategy would involve outwitting Cao Cao by using clever tactics such as forming alliances with other warlords, employing spies and informants to gather information about Cao Cao's plans, and making use of geographical advantages to gain an upper hand in battles. I would also seek to use Cao Cao's own ambition and pride against him, exploiting any weaknesses that could be used to my advantage. Ultimately, the key to victory lies in cleverly utilizing my resources and knowledge to outsmart Cao Cao and outmaneuver him on the battlefield.
    </li>
        </ul>
    </p> 
    <p>
        <ul>
            <li><b>⚪️ NLSOM System (Review):</b>  The NLSOM has successfully solved the user-defined objective by utilizing all agents within the same community. Each agent has provided a unique perspective on how to defeat Cao Cao, and the combination of these perspectives has resulted in a comprehensive understanding of the situation.
    </li>
        </ul>
    </p>  
     <p>
        <ul>
            <li><b>⚪️ NLSOM System (Summary):</b>  GuanYu suggested using both military and diplomatic tactics to defeat Cao Cao. LiuBei suggested relying on the strength of loyal officers, the wisdom of advisers, and the courage of the people. ZhugeLiang suggested using tactical warfare, psychological warfare, and political maneuvering. ZhangFei suggested using strength, courage, intelligence, and the loyalty of followers.
    </li>
        </ul>
    </p> 
     <p>
        <ul>
            <li><b>💰 NLSOM System (Reward):</b>  ["GuanYu": 2, "LiuBei": 2, "ZhugeLiang": 3, "ZhangFei": 2]
    </li>
        </ul>
    </p> 
</details>

### 💾 Usage

#### 1. Install

* Install the dependancies
```
conda env create -n nlsom -f nlsom.yaml
```

```
conda create -n nlsom python=3.8
pip install colorlog==6.7.0
pip install langchain==0.0.158
pip install sqlalchemy==2.0.12
pip install openai
pip install guidance
pip install wolframalpha
pip install wikipedia
pip install arxiv
pip install bs4
pip install streamlit==1.22.0
pip install streamlit_chat==0.0.2.2
pip install colorama
pip install torch==1.13.1
pip install torchvision==0.14.1
pip install transformers
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
pip install easydict
pip install modelscope[cv] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope[nlp] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install fairseq
pip install open_clip_torch
!pip install duckduckgo-search
pip install ffmpeg
pip install trimesh
pip install PyMCubes
pip install scikit-image
pip install TTS
pip install easyocr
pip install guidance





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

* Change the Huggingface/Modelscope save path (Not neccessary but useful)

```bash
>>> import transformers
>>> print(transformers.__file__)
# Get the path: {YOUR_ANACONDA_PATH}/envs/nlsom/lib/python3.8/site-packages/transformers/__init__.py
```

Open the ``{YOUR_ANACONDA_PATH}/envs/nlsom/lib/python3.8/site-packages/transformers/utils/hub.py`` and change the line:
```
torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "{YOUR_NLSOM_PATH}/checkpoints"), "torch"))
hf_cache_home = os.path.expanduser(
   os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "{YOUR_NLSOM_PATH}/checkpoints"), "huggingface"))
)
```

Similarily, change the checkpoints saving place of modelscope,

```bash
>>> import modelscope
>>> print(modelscope.__file__)
# Get the path: ${YOUR_ANACONDA_PATH}/envs/nlsom/lib/python3.8/site-packages/modelscope/__init__.py
```

Open ``{YOUR_ANACONDA_PATH}/envs/nlsom/lib/python3.8/site-packages/modelscope/utils/file_utils.py`` and change the line:
```
default_cache_dir = Path.home().joinpath('{YOUR_NLSOM_PATH}/checkpoints', 'modelscope')
```

```
streamlit run app.py
```




#### 2. Run
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

# API

```
export REPLICATE_API_TOKEN=r8_WssXC8wLfU6nIOZSgm69CM49SFvuObr35zgcu
```

## Preliminary Experiments on Paper 
The original experiments on paper can be found in [experiments](https://github.com/mczhuge/NLSOM/tree/main/experiment). They provide some basic exploration of Mindstorm and natural language-based society of mind.

## Contribute
Please feel free to submit a pull request if you can optimize the identified issues. We will promptly incorporate any improvements.

* Support targets with multiple inputs.
* Support displaying 3D output.
* Add more communities and agents.
* Optimize the prompt of the Mindstorm.
* Design a more accurate reward mechanism.
* Make the NLSOM learnable.

## Acknowledgments

This project utilizes parts of code from the following open-source repositories:

[langchain](https://github.com/hwchase17/langchain), [BabyAGI](https://github.com/yoheinakajima/babyagi), [TaskMatrix](https://github.com/microsoft/TaskMatrix), [DataChad](https://github.com/gustavz/DataChad), [streamlit](https://github.com/streamlit/streamlit).

We also thank great AI platforms and all the used models or APIs:

[huggingface](https://github.com/huggingface/transformers), [modelscope](https://github.com/modelscope/modelscope).



## :black_nib: Citation

References to cite:

```
@article{zhuge2023mindstorms,
  title={Mindstorms in Natural Language-based Societies of Mind},
  author={XXX},
  journal={arXiv preprint arXiv:XXX},
  year={2023}
}

@article{schmidhuber2015learning,
  title={On learning to think: Algorithmic information theory for novel combinations of reinforcement learning controllers and recurrent neural world models},
  author={Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1511.09249},
  year={2015}
}

@book{minsky1988society,
  title={Society of mind},
  author={Minsky, Marvin},
  year={1988},
  publisher={Simon and Schuster}
}
```



