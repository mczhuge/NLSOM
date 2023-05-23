def aokvqa_data(item, split):

    if split != "test":
        image_path = item.split("_!_")[0].split("../aokvqa/datasets/")[1]
        question = item.split("_!_")[1].strip("\n")
        answer_options = eval(item.split("_!_")[2])
        answer_idx = item.split("_!_")[3].strip("\n")
    else:
        image_path = item.split("_!_")[0].split("../aokvqa/datasets/")[1]
        question = item.split("_!_")[2].strip("\n")
        answer_options = eval(item.split("_!_")[3])
        answer_idx = item.split("_!_")[4].strip("\n")

    return image_path, question, answer_options, answer_idx