import streamlit as st
from streamlit_chat import message
from env.recommendation import Organize
from pathlib import Path
import ast
import torch
from colorama import Fore, Back, Style
from PIL import Image

from society.community import *
from env.prompt_bak528 import NLSOM_PREFIX, NLSOM_FORMAT_INSTRUCTIONS, NLSOM_SUFFIX, AI_SOCIETY

# export OPENAI_API_KEY=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
# export OPENAI_API_KEY=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
# eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NDQ4MzEyNywiZXhwIjoxNjg0NTY5NTE2fQ.eyJpZCI6Im1pbmdjaGVuemh1Z2UifQ.Kmm8XriNPdc_a4fF2tLOeGH2SvlpNQ84YT2ZiDaK0G9CJ9W8DmXuBhWb8dsTyTJIweCDg7pfy2-ko8b5u7C46Q
# mczhuge

#openai.api_key = os.getenv('OPENAI_API_KEY')
os.makedirs('data', exist_ok=True)

from constants import (
    ACTIVELOOP_HELP,
    APP_NAME,
    AUTHENTICATION_HELP,
    CHUNK_SIZE,
    DEFAULT_DATA_SOURCE,
    ENABLE_ADVANCED_OPTIONS,
    FETCH_K,
    MAX_TOKENS,
    OPENAI_HELP,
    PAGE_ICON,
    REPO_URL,
    TEMPERATURE,
    USAGE_HELP,
    K,
)

from utils import (
    #advanced_options_form,
    authenticate,
    delete_uploaded_file,
    generate_response,
    generate_reward,
    logger,
    save_uploaded_file,
    update_chain,
)


# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)

LOGO_FILE = os.path.join("config", "nlsom.png")
st.title(':orange[Mindstorms] in NL:blue[SOM]')
st.text("1️⃣ Enter your API keys.  ")
st.text("2️⃣ Next, upload a file and your task. ")
st.text("3️⃣ I will automatically organize an NLSOM and solve the task.")


SESSION_DEFAULTS = {
    "past": [],
    "usage": {},
    "device": "cuda:0", # TODO: support multiple GPUs
    "chat_history": [],
    "generated": [],
    "data_name": [],
    "language": "English",
    "models": {},
    "communities": {},
    "agents": {},
    "load_dict": {},
    "data_source": [], #DEFAULT_DATA_SOURCE,
    "uploaded_file": None,
    "auth_ok": False,
    "openai_api_key": None,
    "activeloop_token": None,
    "activeloop_org_name": None,
    "k": K,
    "fetch_k": FETCH_K,
    "chunk_size": CHUNK_SIZE,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
}


# Initialise session state variables
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Sidebar with Authentication
# Only start App if authentication is OK
with st.sidebar:

    st.title("🔗 API Login", help=AUTHENTICATION_HELP)
    # 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
    with st.form("authentication"):
        openai_api_key = st.text_input(
            "🕹 OpenAI API Key1",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        openai_api_key2 = st.text_input(
            "🕹 OpenAI API Key2",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        openai_api_key3 = st.text_input(
            "🕹 OpenAI API Key3",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        openai_api_key4 = st.text_input(
            "🕹 OpenAI API Key4",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )


        language = st.selectbox(
        "📖 Language",
        ('English', '中文'))

        st.session_state["language"] = language
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            #authenticate(openai_api_key, activeloop_token, activeloop_org_name)
            authenticate(openai_api_key)
    
    REPO_URL = "https://github.com/mczhuge/AutoMind"
    st.info(f"🟢 Github Page: [KAUST-AINT-NLSOM]({REPO_URL})")
    st.image(LOGO_FILE)
    if not st.session_state["auth_ok"]:
        st.stop()

    # Clear button to reset all chat communication
    clear_button = st.button("Clear Conversation", key="clear")

    # Advanced Options
    # if ENABLE_ADVANCED_OPTIONS:
    #     advanced_options_form()



# the chain can only be initialized after authentication is OK
if "chain" not in st.session_state:
    update_chain()

if clear_button:
    # resets all chat history related caches
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["chat_history"] = []


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
    print(dir_path)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == "agent.py": #file.endswith('.py'):
                file_path = os.path.join(root, file)
                classes = get_agent_class(file_path)
                #results.append((file_path, classes))
                results.append(classes)
    return results[0]


def load_candidate(candidate_list, AI_SOCIETY):

    #candidate_list = candidate_list.replace("[", "").replace("]", "").replace("'", "").split(",")

    print("==candidate_list==", candidate_list)

    device = st.session_state["device"]

    load_dict = {}
    for community in candidate_list:
        print("==community==", community.strip())
        agents = traverse_dir(community.strip())
        for agent in agents:
            print("==community, agent==", community.strip(), agent)
            # Change the device "cuda:0" if you have more.
            st.session_state["load_dict"][agent] = device #"cpu" #"cuda:0" # TODO: Automatically load into different GPUs
            if str(community).strip() not in st.session_state["agents"].keys():
                st.session_state["agents"][str(community).strip()] = [agent]
            else:
                st.session_state["agents"][str(community).strip()].append(agent)

    print("agents:", st.session_state["agents"])
    st.session_state["generated"].append("最终加载的AI社区和相应的代理有:\n{}".format(st.session_state["agents"]))
    
    st.session_state["chat_history"].append("最终加载的AI社区和相应的代理有:\n{}".format(st.session_state["agents"]))
    print(Fore.BLUE + "最终加载的AI社区和相应的代理有:\n{}".format(st.session_state["agents"]), end='')
    print(Style.RESET_ALL)
    #print("==self.load_dict==", st.session_state["load_dict"])
    for class_name, device in st.session_state["load_dict"].items():
        print("+++",class_name, device)
        st.session_state["models"][class_name] = globals()[class_name](device=device)
    
    #print("==self.models==", st.session_state["models"])
        
    st.session_state["tools"] = []
    for instance in st.session_state["models"].values():
        print("==工具==", st.session_state["models"].values())
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)
                st.session_state["tools"].append(Tool(name=func.name, description=func.description, func=func))

    print("工具:", st.session_state["tools"])


# generate new chain for new data source / uploaded file
# make sure to do this only once per input / on change

if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:

    print("到达这一步1")
    logger.info(f"Uploaded file: '{uploaded_file.name}'")
    st.session_state["uploaded_file"] = uploaded_file
    data_source = save_uploaded_file(uploaded_file)
    #data_name = st.session_state["data_name"] = "data/" + data_source.split("/")[-1]
    filename = "data/" + uploaded_file.name

    # TODO: 识别上传图片的属性

    if len(re.findall(r'\b([-\w]+\.(?:jpg|png|jpeg|bmp|svg|ico|tif|tiff|gif|JPG))\b', filename)) != 0:
        filetype = "image"
        img = Image.open(filename)
        width, height = img.size
        ratio = min(512/ width, 512/ height)
        img = img.resize((round(width * ratio), round(height * ratio)))
        img = img.convert('RGB')
        img.save(filename, "PNG")

    #data_name = st.session_state["data_name"] = f"![](file={filename})*{filename}*"
    data_name = st.session_state["data_name"] = filename
    print("到达这一步2")
    if st.session_state["language"] == "English":
        st.session_state["generated"].append(f"Receive a file, it stored in {data_name}")
        #st.session_state["chat_history"].append((data_name, f"Receive a file, it stored in {data_name}"))
    else:
        st.session_state["generated"].append(f"收到一个文件, 它存储在如下路径: {data_name}")
    st.session_state["chat_history"].append((data_name, f"Receive a file, it stored in {data_name}"))
    
    print("到达这一步3", st.session_state["chat_history"])
    st.session_state["data_source"] = data_source
    update_chain()
    #delete_uploaded_file(uploaded_file)

# container for chat history
response_container = st.container()
# container for text box
container = st.container()



# As streamlit reruns the whole script on each change
# it is necessary to repopulate the chat containers
with container:
    with st.form(key="prompt_input", clear_on_submit=True):
        user_input = st.text_area("🎯 Your target:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:

        st.session_state["past"].append(user_input)
        print("1", st.session_state["data_name"])
        print("2", user_input)
        community = Organize(user_input)
        if st.session_state["data_name"] != []:
            #community = Organize(st.session_state["data_name"] + " " + user_input) # TODO: 验证是否有识别这张图
            user_input = st.session_state["data_name"] + ", " + user_input
        #message(f"基于这个目标, 我推荐NLSOM需要包含以下团体: {community}")
        community = community.replace("[", "").replace("]", "").replace("'", "").split(",")
        num_icon = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
        recommendation = "\n"
        for i in range(len(community)):
            recommendation += (num_icon[i] + community[i]) + "\n"
        st.session_state["generated"].append(f"基于这个目标, 我推荐NLSOM需要包含以下AI社区: {recommendation}")
        print(Fore.BLUE + f"基于这个目标, 我推荐NLSOM需要包含以下AI社区: {recommendation}", end='')
        print(Style.RESET_ALL)
        st.session_state["chat_history"].append(f"基于这个目标, 我推荐NLSOM需要包含以下AI社区: {recommendation}")
        load_candidate(community, AI_SOCIETY)

        print("===初始记忆池子===",  st.session_state["chat_history"])
        responce = generate_response(user_input, st.session_state["tools"], st.session_state["chat_history"])
        #reward = generate_reward(user_input, st.session_state["communities"], st.session_state["agents"], st.session_state["chat_history"])
        review, output, reward = responce.split("\n")[0], responce.split("\n")[1], responce.split("\n")[2]
        if "Analyze the employed agents" in review: # The review was unsuccessful, possibly due to the ongoing process or the brevity of the content.
            review = review.split("Analyze the employed agents")[0].strip("[").strip("]")
        
        st.session_state["generated"].append(review)
        st.session_state["generated"].append(output)
        st.session_state["generated"].append(reward)

        st.session_state["generated"].append(responce)
        #st.session_state["generated"].append(reward)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["past"])):
            print(st.session_state["past"])
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        for i in range(len(st.session_state["generated"])):
            print(st.session_state["generated"])
            message(st.session_state["generated"][i], key=str(i))

            image_parse = re.findall(r'\b([-\w]+\.(?:jpg|png|jpeg|bmp|svg|ico|tif|tiff|gif|JPG))\b', st.session_state["generated"][i])
            if image_parse != []:
                print("图像名字: ", image_parse)
                image = Image.open(os.path.join("data", image_parse[-1]))
                st.image(image, caption=image_parse[-1])

        # TODO: Reward


        
# Usage sidebar with total used tokens and costs
# We put this at the end to be able to show usage starting with the first response
with st.sidebar:
    if st.session_state["usage"]:
        st.divider()
        st.title("Usage", help=USAGE_HELP)
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
        col2.metric("Total Costs in $", st.session_state["usage"]["total_cost"])




