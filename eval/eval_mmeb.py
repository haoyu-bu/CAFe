import json
import sys
from dataclasses import dataclass

from VLM2Vec.src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, ProcessorMixin
import transformers

from VLM2Vec.src.dataset import EvalDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from VLM2Vec.evaluation.eval_utils import get_pred

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.train.train import preprocess_qwen, preprocess_multimodal

# FashionIQ Wiki-SS-NQ OVEN EDIS MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing
# VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA
# OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA VizWiz GQA TextVQA
# ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211 

@dataclass
class EvalLlavaCollator:
    config: transformers.PretrainedConfig
    processor: ProcessorMixin
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def _get_batch_inputs(self, examples):
        inputs = [[[{"from": "human", "value": e[0].replace("<|image_1|>", "<image>")}, {"from": "assistant", "value": e[0].replace("<|image_1|>", "<image>")}]] for e in examples]
        if examples[0][1] is None and "<|image_1|>" in examples[0][0]:
            inputs = [[[{"from": "human", "value": e[0].replace("<|image_1|>", "")}, {"from": "assistant", "value": e[0].replace("<|image_1|>", "")}]] for e in examples]
        instances = [preprocess_qwen(i, self.tokenizer, has_image=True, multimodal_input=True) for i in inputs]
        input_ids, labels, input_text_ids, input_image_ids = tuple([instance[key][0] for instance in instances] for key in ("input_ids", "labels", "input_text_ids", "input_image_ids"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        input_text_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_text_ids]
        input_image_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_image_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_text_ids = self.pad_sequence(input_text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_image_ids = self.pad_sequence(input_image_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), 
                     input_text_ids=input_text_ids, text_attention_mask=input_text_ids.ne(self.tokenizer.pad_token_id), input_image_ids=input_image_ids, image_attention_mask=input_image_ids.ne(self.tokenizer.pad_token_id))

        if examples[0][1] is not None: # has image
            images = [process_images([f], self.processor, self.config) for _, f in examples]
            images = [im for im_list in images for im in im_list]
            batch["image_sizes"] = [f.size for _, f in examples]
            batch["modalities"] = ["image" for im in images]
            batch["images"] = images

        return batch

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    pretrained = "/CKPT/PATH/"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "cuda"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation=None, device_map=device_map)  
    conv_template = "qwen_1_5"
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalLlavaCollator(
        config=model.config,
        processor=image_processor,
        tokenizer=tokenizer,
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            print("qry and tgt encoded, skipping")
            continue

        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                batch_device = {key: value.to(training_args.device) for key, value in batch.items() if not isinstance(value, list)}
                if "images" in batch:
                    batch_device["images"] = [v.to(training_args.device) for v in batch["images"]]
                    batch_device["image_sizes"] = batch["image_sizes"]
                    batch_device["modalities"] = batch["modalities"]
                batch = batch_device
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    if "images" in batch:
                        text_embeds, image_embeds = model(
                            input_ids=batch["input_ids"],
                            input_text_ids=batch["input_text_ids"],
                            input_image_ids=batch["input_image_ids"],
                            text_attention_mask=batch["text_attention_mask"],
                            image_attention_mask=batch["image_attention_mask"],
                            images=batch["images"],
                            image_sizes=batch["image_sizes"],
                            modalities=batch["modalities"],
                            output_hidden_states=True,
                            compute_embedding=True,
                            language_generation=False,
                            emb_type='last',
                            emb_head=False,
                        )
                        output = image_embeds
                    else:
                        text_embeds, image_embeds = model(
                            input_ids=batch["input_ids"],
                            input_text_ids=batch["input_text_ids"],
                            input_image_ids=batch["input_image_ids"],
                            text_attention_mask=batch["text_attention_mask"],
                            image_attention_mask=batch["image_attention_mask"],
                            output_hidden_states=True,
                            compute_embedding=True,
                            language_generation=False,
                            emb_type='last',
                            emb_head=False,
                        )
                        output = text_embeds
                encoded_tensor.append(output.cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch_device = {key: value.to(training_args.device) for key, value in batch.items() if not isinstance(value, list)}
                if "images" in batch:
                    batch_device["images"] = [v.to(training_args.device) for v in batch["images"]]
                    batch_device["image_sizes"] = batch["image_sizes"]
                    batch_device["modalities"] = batch["modalities"]
                batch = batch_device
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    if "images" in batch:
                        text_embeds, image_embeds = model(
                            input_ids=batch["input_ids"],
                            input_text_ids=batch["input_text_ids"],
                            input_image_ids=batch["input_image_ids"],
                            text_attention_mask=batch["text_attention_mask"],
                            image_attention_mask=batch["image_attention_mask"],
                            images=batch["images"],
                            image_sizes=batch["image_sizes"],
                            modalities=batch["modalities"],
                            output_hidden_states=True,
                            compute_embedding=True,
                            language_generation=False,
                            emb_type='last',
                            emb_head=False,
                        )
                        output = image_embeds
                    else:
                        text_embeds, image_embeds = model(
                            input_ids=batch["input_ids"],
                            input_text_ids=batch["input_text_ids"],
                            input_image_ids=batch["input_image_ids"],
                            text_attention_mask=batch["text_attention_mask"],
                            image_attention_mask=batch["image_attention_mask"],
                            output_hidden_states=True,
                            compute_embedding=True,
                            language_generation=False,
                            emb_type='last',
                            emb_head=False,
                        )
                        output = text_embeds
                encoded_tensor.append(output.cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,
        )
        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")


if __name__ == "__main__":
    main()
