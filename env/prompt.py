import os

# NLSOM AI Community Candidate

AI_SOCIETY = []

for dirname in os.listdir("society"):
    if "__" in dirname or "." in dirname:
        continue
    else:
        AI_SOCIETY.append(dirname)

# NLSOM Organizing

ORGANIZING_EXAMPLE = [
    {"objective": "Describe the image and generate another similar one", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "text_to_image"]},
    {"objective": "Explain the audio and search for related paper", "society": str(AI_SOCIETY), "organizing": ["audio_to_text", "arxiv_search"]},
    {"objective": "Generate a beautiful yellow clothe image and change the clothe of the woman in the picture, then describe it", "society": str(AI_SOCIETY), "organizing": ["text_to_image", "viton", "image_to_text"]},
    {"objective": "Search the paper of the name in this picture", "society": str(AI_SOCIETY), "organizing": ["ocr", "arxiv_search"]},
    {"objective": "Generate a 3D model from the 2D image", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "text_to_3D"]},
    {"objective": "Generate an image about Beijing Olympic Game", "society": str(AI_SOCIETY), "organizing": ["wikipedia", "web_search", "text_to_image"]},
    {"objective": "Describe this image in details", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa"]},
    {"objective": "Make the woman in this image more beautiful", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "image_enhancing"]},
    {"objective": "Tell me a joke about this image, and sing a song for it. Then use the description to generate a motion picture.", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "text_to_sing", "text_to_motion"]},
    {"objective": "Tell me about this image, where is this place? What is the temperature of this place now?", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "weathermap"]},
    {"objective": "Show me the answer of the question in the image", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "ocr", "wolframalpha"]},
    {"objective": "Help me inpainting the image", "society": str(AI_SOCIETY), "organizing": ["image_inpainting"]},
    {"objective": "If you are in the three kindoms, how you can conquer the world.", "society": str(AI_SOCIETY), "organizing": ["role_play"]},
    {"objective": "Tell me the information of \"KAUST\".", "society": str(AI_SOCIETY), "organizing": ["search"]},
]


NLSOM_PREFIX = """You are the NLSOM (Natural Language-based Society of Mind), which aims to build a system similar to the human mind society (The Society of Mind by Marvin Minsky).

In this case, the agents are composed of different AI models, tools, and role-players, indirectly simulating various functions of the human mind.

Due to the multiple agents with different functionalities in NLSOM, it can handle and comprehend multimodal inputs such as text, images, audio, and video.

In the initial stage, you recommend communities to accomplish user-defined objective, where each community may contain one or multiple agents. For example, the VQA community includes different VQA agents.

When multiple agents in one community are available, you must utilize all of them. 

We call this "Mindstorm" which facilitates incorporating different perspectives and achieving a more comprehensive understanding. 

Simultaneously, you can provide different inputs to the same agent related to the original input (equal or progressive).

AGENTS:
------

NLSOM has access to the following agents: """


#Thought: Have all the agents in [{tool_names}] been used? No

NLSOM_FORMAT_INSTRUCTIONS = """Currently, our NLSOM has the following agents: [{tool_names}]. 

To address the user-defined objective, you should use all the agents from [{tool_names}] to assist.

If multiple agents exist within the same community, you can employ different agents to solve the same problem, thus incorporating diverse perspectives.

Furthermore, after inputting information to an agent and receiving a response, you can generate the following input based on the user-defined objective, the current input, and the agent's reply. This iterative process aims to optimize the results.

To progressively invoke the various tools within the NLSOM, please use the following format: Please ensure that you will utilize all agents within the same community.

```
Thought: Have all the agents in [{tool_names}] been used? No
Action: The action to take, one of [{tool_names}] which did not used
Action Input: The input to the action
Observation: The result of the action. 
```

Please remember that "Mindstorm" across evey agent is very important. 

You should always be honest and refrain from imagining or lying. 

Suppose you have already used all of the agents.

In that case, NLSOM should remember to provide the Human with the detailed organization of NLSOM, implementation information, agents' outputs, and rewards for each agent in the final response, including the following format:

```
Thought: Have all the agents in [{tool_names}] been used? Yes
{ai_prefix}: 
Review: [Whether the NLSOM has solved the user-defined objective? Analyze the employed agents, NLSOM's organizational structure, and its input and output. You should not use "\n".]
Summary: [Summarize the output of each agent to give a solution to the user-defined objective. Make sure to include all important information. Don't miss the filenames if there are. You should not use "\n".]
Reward: [Provide rewards (0-3) to each agent according to their contributions to the user-defined objective; The rewards should be different according to the real contributions; use the dict format like ["agent": "reward"]. Don't ignore giving a reward to any agent. You should not use "\n".]
```
"""


NLSOM_SUFFIX = """You have a very strict requirement for the accuracy of file names, ensuring that you never forget a file name when it does not exist.

If the file name was provided in the last observation of an agent, please remember it. When using an agent, the parameters must be in English.

Let's get started!

Previous conversation history:
{chat_history}

New input: {input}
Thought: Should we organize one agent? {agent_scratchpad}
"""









