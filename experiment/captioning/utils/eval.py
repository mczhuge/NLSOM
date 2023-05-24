from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

########################
## Sentence Similarity
########################
def similariry_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def caculate_similarity(results, data, model):
    scores = []
    """
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()

        score = similariry_score(target, prediction, model)
        scores.append(score)
    """

    score = similariry_score(data, results, model)
    scores.append(score)



    avg_score = sum(scores) / len(scores)
    return avg_score




def evaluate_cap(gt, generation):


    ## Similarity
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    similarity = caculate_similarity(generation, gt, model)

    scores = {
            "captioning":{
                'similarity': similarity * 100,
            }
            }
    
    return scores