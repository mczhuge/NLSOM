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
            <li><b>🔴 User:</b> Tell me the details about "AGI"?</li>
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
            <li><b>🟢 GuanYu:</b> To defeat Cao Cao, I would use my strategic acumen and superior martial prowess to lead an army of loyal followers to take a decisive victory. My strength and courage are unparalleled, and I have never been known to back down from a challenge. With my inspired leadership and unwavering commitment to justice, we can bring Cao Cao to justice and restore peace and stability to the Three Kingdoms.
    </li>
        </ul>
    </p>  
    <p>
        <ul>
            <li><b>🟡 LiuBei:</b> The most effective approach to defeat Cao Cao in the Three Kingdoms period is to employ a combination of tactical strategies. Firstly, I would use my political prowess to unify the other warlords and form an alliance that would be strong enough to oppose Cao Cao. Secondly, I would take advantage of my highly organized military tactics and strategies to effectively weaken Cao Cao's army. Finally, I would ensure that my own troops are well fed and well-armed, and that they maintain morale. By combining these tactical measures, I am confident that I can defeat Cao Cao and ultimately bring peace to the Three Kingdoms period.
    </li>
        </ul>
    </p>   
    <p>
        <ul>
            <li><b>🔵 ZhugeLiang:</b>  My approach to defeating Cao Cao would be to outwit him through the use of strategy and cunning. I believe that the ability to outthink and outsmart your enemies is more powerful than any physical force. I have honed my skills in the art of war and I have a deep understanding of the dynamics of the Three Kingdoms period. I would use this knowledge to my advantage and create a strategy that would take Cao Cao by surprise. I would use guerilla tactics to surprise him and break his momentum. I would also use deception to mislead him and cause confusion. Ultimately, I believe that the key to victory is the combination of strategy, intelligence, and courage.
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
            <li><b>⚪️ NLSOM System (Review):</b>  All four agents have been used to answer the user's question. GuanYu suggested using both military and diplomatic tactics, LiuBei suggested using a combination of military and diplomatic strategies, ZhugeLiang suggested using tactical warfare, psychological warfare, and political maneuvering, and ZhangFei suggested using a combination of strategy and guile.
    </li>
        </ul>
    </p>  
     <p>
        <ul>
            <li><b>⚪️ NLSOM System (Output):</b>  The best way to defeat Cao Cao is to use a combination of military and diplomatic tactics, strategic alliances, and clever tactics such as forming alliances with other warlords, employing spies and informants, and making use of geographical advantages.
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
            <li><b>⚪️ NLSOM System (Review):</b>  All four agents have been used to answer the user's question. GuanYu suggested using both military and diplomatic tactics, LiuBei suggested using a combination of military and diplomatic strategies, ZhugeLiang suggested using tactical warfare, psychological warfare, and political maneuvering, and ZhangFei suggested using a combination of strategy and guile.
    </li>
        </ul>
    </p>  
     <p>
        <ul>
            <li><b>⚪️ NLSOM System (Output):</b>  The best way to defeat Cao Cao is to use a combination of military and diplomatic tactics, strategic alliances, and clever tactics such as forming alliances with other warlords, employing spies and informants, and making use of geographical advantages.
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



