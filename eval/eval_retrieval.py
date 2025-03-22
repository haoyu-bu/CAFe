from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

import numpy as np
import pandas as pd

from datasets import load_dataset
import torch.nn.functional as F

from llava.train.train import preprocess_qwen

import argparse
parser = argparse.ArgumentParser(description='Zero-shot Retrieval Evaluation')
parser.add_argument('--ckpt', type=str, help='path of checkpoint')
parser.add_argument('--emb', type=str, default="last", help='embedding type')
parser.add_argument('--emb_head', action='store_true', help='if add embedding head')
args = parser.parse_args()

flickr_test = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval")
coco_test = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

warnings.filterwarnings("ignore")
pretrained = args.ckpt
model_name = "llava_qwen"
device = "cuda"
device_map = "cuda"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,attn_implementation=None,  device_map=device_map)  # Add any other thing you want to pass in llava_model_args
emb_type = args.emb
emb_head = args.emb_head
print(model.config)

model.eval()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def eval(dataset, task_name, embedding='last', recall_k_list=[1, 5, 10]):
    print("dataset" + task_name, "embedding: " + embedding)
    img_embs = []
    txt_embs = []
    texts_image_index = []
    start = 0
    for x in dataset["test"]:
        end = start + 1
        inds = torch.arange(start, end)
        start = end
        image = x['image']
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        question =  DEFAULT_IMAGE_TOKEN + "\nCompress this image in one word:"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "")
        prompt_question = conv.get_prompt() + '<|im_end|>'

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        text_embeds, image_embeds = model(
            input_ids=input_ids,
            input_text_ids=input_ids,
            input_image_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            output_hidden_states=True,
            compute_embedding=True,
            language_generation=False,
            emb_type=emb_type,
            emb_head=emb_head,
        )

        img_embs.append(image_embeds.detach().to('cpu'))
        del text_embeds, image_embeds

        batch_texts_image_index = [ind for ind, texts in zip(inds, [x['caption']]) for text in texts]
        for caption in x['caption']:
            question = caption + "Compress this sentence in one word:"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], "")
            prompt_question = conv.get_prompt() + '<|im_end|>'
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            text_embeds, image_embeds = model(
                input_ids=input_ids,
                input_text_ids=input_ids,
                input_image_ids=input_ids,
                output_hidden_states=True,
                compute_embedding=True,
                language_generation=False,
                emb_type=emb_type,
                emb_head=emb_head,
            )

            txt_embs.append(text_embeds.detach().to('cpu'))
            del text_embeds, image_embeds
        
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(img_embs[0])
    images_emb = torch.cat(img_embs).type(torch.float32)
    texts_emb = torch.cat(txt_embs).type(torch.float32)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    print(metrics)
    return metrics

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

eval(flickr_test, 'flickr', 'last')
eval(coco_test, 'coco', 'last')
