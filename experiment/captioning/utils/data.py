def tara_data(item, split):


    image_path = item.split("_!_")[3]
    image_abstract = item.split("_!_")[0]
    image_paragraph = item.split("_!_")[1]
    ner = item.split("_!_")[2].strip("\n")
    place = item.split("_!_")[4].strip("\n")


    return image_path, image_abstract, image_paragraph, ner, place