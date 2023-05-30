import streamlit as st
from streamlit_chat import message
from env.recommendation import Organize
from pathlib import Path
import ast
import re
from colorama import Fore, Style
from PIL import Image
import os
import soundfile as sf
import shutil
import torch

from society.community import *
from env.prompt import AI_SOCIETY

from langchain.agents.tools import Tool

os.makedirs('data', exist_ok=True)

from setting import (
    APP_NAME,
    AUTHENTICATION_HELP,
    OPENAI_HELP,
    HUGGINGFACE_HELP,
    BINGSEARCH_HELP,
    WOLFRAMALPHA_HELP,
    REPLICATE_HELP,
    PAGE_ICON,
    REPO_URL,
    USAGE_HELP,
)

from utils import (
    authenticate,
    delete_uploaded_file,
    generate_response,
    logger,
    save_uploaded_file,
)


# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)

LOGO_FILE = os.path.join("assets", "nlsom.png")
st.title(':orange[Mindstorms] in NL:blue[SOM]')
st.text("1️⃣ Enter API keys.")
st.text("2️⃣ Upload the task/file. ")
st.text("3️⃣ System organize an NLSOM and conduct mindstorms.")
st.text("4️⃣ Sovle the task.")


SESSION_DEFAULTS = {
    "past": [],
    "usage": {},
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "chat_history": [],
    "generated": [],
    "data_name": [],
    "language": "English",
    "models": {},
    "communities": {},
    "agents": {},
    "load_dict": {},
    "data_source": [], 
    "uploaded_file": None,
    "auth_ok": False,
    "openai_api_key": None,
    "huggingface_api_key": None,
    "bingsearch_api_key": None,
    "wolframalpha_api_key": None,
    "replicate_api_key": None,
}


# Initialise session state variables
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Move .env to .streamlit/secrets.toml
os.makedirs(".streamlit", exist_ok=True)
shutil.copyfile(".env", ".streamlit/secrets.toml")

# Sidebar with Authentication
# Only start App if authentication is OK
with st.sidebar:

    st.title("🔗 API Pool", help=AUTHENTICATION_HELP)
    with st.form("authentication"):

        openai_api_key = st.text_input(
            "🕹 OpenAI API",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        huggingface_api_key = st.text_input(
            "🕹 HuggingFace API",
            type="password",
            help=HUGGINGFACE_HELP,
            placeholder="This field is optional",
        )
        bingsearch_api_key = st.text_input(
            "🕹 BingSearch API",
            type="password",
            help=BINGSEARCH_HELP,
            placeholder="This field is optional",
        )
        wolframalpha_api_key = st.text_input(
            "🕹 WolframAlpha API",
            type="password",
            help=WOLFRAMALPHA_HELP,
            placeholder="This field is optional",
        )
        replicate_api_key = st.text_input(
            "🕹 Replicate API",
            type="password",
            help=REPLICATE_HELP,
            placeholder="This field is optional",
        )

        language = st.selectbox(
        "📖 Language",
        ('English', '中文'))

        st.session_state["language"] = language
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            #authenticate(openai_api_key, activeloop_token, activeloop_org_name)
            authenticate(openai_api_key)
    
    REPO_URL = "https://github.com/AI-Initiative-KAUST/NLSOM"
    st.info(f"🟢 Github Page: [KAUST-AINT-NLSOM]({REPO_URL})")
    st.image(LOGO_FILE)
    if not st.session_state["auth_ok"]:
        st.stop()

    # Clear button to reset all chat communication
    clear_button = st.button("Clear Conversation", key="clear")

if clear_button:
    # Resets all chat history related caches
    # delete_uploaded_file(st.session_state["data_source"])
    st.session_state["past"] = []
    st.session_state["usage"] = {}
    st.session_state["generated"] = []
    st.session_state["chat_history"] = []
    st.session_state["data_name"] = []
    st.session_state["models"] = {}
    st.session_state["communities"] = {}
    st.session_state["agents"] = {}
    st.session_state["load_dict"] = {}
    st.session_state["data_source"] = []
    st.session_state["uploaded_file"] = None

# file upload and data source inputs
uploaded_file = st.file_uploader("Upload a file")
data_source = st.text_input(
    "Enter any data source",
    placeholder="Any path or URL pointing to a file",
)

def get_agent_class(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    classes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            name = node.name
            classes.append(name)
    return classes


def traverse_dir(community):
    results = []
    dir_path = "./society/"+community+"/"
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == "agent.py": #file.endswith('.py'):
                file_path = os.path.join(root, file)
                classes = get_agent_class(file_path)
                results.append(classes)
    return results[0]


def load_candidate(candidate_list, AI_SOCIETY):


    device = st.session_state["device"]
    print(f"Current device: {device}")

    for community in candidate_list:
        agents = traverse_dir(community.strip())
        for agent in agents:
            st.session_state["load_dict"][str(agent)] = device 
            if str(community).strip() not in st.session_state["agents"].keys():
                st.session_state["agents"][str(community).strip()] = [str(agent)]
            else:
                st.session_state["agents"][str(community).strip()].append(str(agent))

    st.session_state["generated"].append("We load the recommended AI communities with their their corresponding agents:\n{}".format(st.session_state["agents"]))
    
    st.session_state["chat_history"].append("We load the recommended AI communities with their their corresponding agents:\n{}".format(st.session_state["agents"]))
    print(Fore.BLUE + "We load the recommended AI communities with their their corresponding agents:\n{}".format(st.session_state["agents"]), end='')
    print(Style.RESET_ALL)
    for class_name, device in st.session_state["load_dict"].items():
        st.session_state["models"][class_name] = globals()[class_name](device=device)
    
    st.session_state["tools"] = []
    for instance in st.session_state["models"].values():
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)
                st.session_state["tools"].append(Tool(name=func.name, description=func.description, func=func))




# Only support one file currently

if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:

    logger.info(f"Uploaded file: '{uploaded_file.name}'")
    st.session_state["uploaded_file"] = uploaded_file
    data_source = save_uploaded_file(uploaded_file)
    filename = "data/" + uploaded_file.name

    # image source
    if len(re.findall(r'\b([-\w]+\.(?:jpg|png|jpeg|bmp|svg|ico|tif|tiff|gif|JPG))\b', filename)) != 0:
        filetype = "image"
        img = Image.open(filename)
        width, height = img.size
        ratio = min(512/ width, 512/ height)
        img = img.resize((round(width * ratio), round(height * ratio)))
        img = img.convert('RGB')
        img.save(filename, "PNG")

    # audio source
    if len(re.findall(r'\b([-\w]+\.(?:wav|flac|mp3))\b', filename)) != 0:
        filetype = "audio"

    # video source
    if len(re.findall(r'\b([-\w]+\.(?:avi|mov|flv|mp4|wmv))\b', filename)) != 0:
        filetype = "video"

    #data_name = st.session_state["data_name"] = f"![](file={filename})*{filename}*"
    data_name = st.session_state["data_name"] = filename
    st.session_state["generated"].append(f"Receive a {filetype} file, it stored in {data_name}")
    st.session_state["chat_history"].append((data_name, f"Receive the {filetype} file, it stored in {data_name}"))
    st.session_state["data_source"] = data_source
    

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

# As streamlit reruns the whole script on each change
# it is necessary to repopulate the chat containers
with container:
    with st.form(key="prompt_input", clear_on_submit=True):
        user_input = st.text_area("🎯 Your task:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:

        st.session_state["past"].append(user_input)
        community = Organize(user_input)
        if st.session_state["data_name"] != []:
            user_input = st.session_state["data_name"] + ", " + user_input
        print(Fore.BLUE + f"User Input: {user_input}", end='')
        print(Style.RESET_ALL)
        community = community.replace("[", "").replace("]", "").replace("'", "").split(",")
        num_icon = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
        recommendation = "\n"
        for i in range(len(community)):
            recommendation += (num_icon[i] + community[i]) + "\n"
        st.session_state["generated"].append(f"Based on this objective, I recommend that NLSOM includes the following AI communities: {recommendation}")
        print(Fore.BLUE + f"Based on this objective, I recommend that NLSOM includes the following AI communities: {recommendation}", end='')
        print(Style.RESET_ALL)
        st.session_state["chat_history"].append(f"Based on this objective, I recommend that NLSOM includes the following AI communities: {recommendation}")
        load_candidate(community, AI_SOCIETY)

        responce = generate_response(user_input, st.session_state["tools"], st.session_state["chat_history"])

        print("###"*20)
        print(responce)
        print("###"*20)

        review, output, reward = responce.split("\n")[0], responce.split("\n")[1], responce.split("\n")[2]
        if "Analyze the employed agents" in review: # The review was unsuccessful, possibly due to the ongoing process or the brevity of the content.
            review = review.split("Analyze the employed agents")[0].strip("[").strip("]")
        
        st.session_state["generated"].append(review)
        st.session_state["generated"].append(output)
        st.session_state["generated"].append(reward)
        st.session_state["generated"].append(responce)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["past"])):
            #print(st.session_state["past"])
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        for i in range(len(st.session_state["generated"])):
            #print(st.session_state["generated"])
            if i==0: continue
            message(st.session_state["generated"][i], key=str(i))

            image_parse = re.findall(r'\b([-\w]+\.(?:jpg|png|jpeg|bmp|svg|ico|tif|tiff|gif|JPG))\b', st.session_state["generated"][i])
            if image_parse != []:
                image = Image.open(os.path.join("data", image_parse[-1]))
                st.image(image, caption=image_parse[-1])

            audio_parse = re.findall(r'\b([-\w]+\.(?:wav|flac|mp3))\b', st.session_state["generated"][i])
            if audio_parse != []:
                audio_format = audio_parse[-1].split(".")[-1]
                
                if audio_format != "wav":
                    audio_bytes, samplerate = sf.read(os.path.join("data", audio_parse[-1]))
                    audio_format = "wav"
                    st.audio(audio_bytes, format=f"audio/{audio_format}", sample_rate=samplerate)
                else:
                    audio_file = open(os.path.join("data", audio_parse[-1]))
                    audio_bytes = audio_file.read(os.path.join("data", audio_parse[-1]))
                    st.audio(audio_bytes, format=f"audio/{audio_format}")

            video_parse = re.findall(r'\b([-\w]+\.(?:avi|mov|flv|mp4|wmv))\b', st.session_state["generated"][i])
            if video_parse != []:
                video_file = open(os.path.join("data", video_parse[-1]), "rb")
                video_bytes = video_file.read()
                st.video(video_bytes)


# Usage sidebar with total used tokens and costs
# We put this at the end to be able to show usage starting with the first response
with st.sidebar:
    if st.session_state["usage"]:
        st.divider()
        st.title("Usage", help=USAGE_HELP)
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
        col2.metric("Total Costs in $", st.session_state["usage"]["total_cost"])
