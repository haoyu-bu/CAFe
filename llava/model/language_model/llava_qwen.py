#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.img_emb_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.text_emb_head = nn.Linear(config.hidden_size, config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        input_image_ids: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        language_generation: Optional[bool] = True,
        compute_embedding: Optional[bool] = False,
        emb_type: Optional[str] = 'last',
        emb_head: Optional[bool] = False,
        multimodal_input: Optional[bool] = False,
        cache_position=None,
        image_only=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if language_generation and inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            if language_generation:
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            if compute_embedding:
                def extract_emb(input_ids, position_ids, attention_mask, past_key_values=None, labels=None, images=None, modalities=None, image_sizes=None):
                    (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                    hidden_states = outputs[0]
                    # last token
                    if emb_type == 'last':
                        if attention_mask is not None:
                            sequence_lengths = attention_mask.sum(dim=1) - 1
                            batch_size = hidden_states.shape[0]
                            return hidden_states[
                                    torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
                        else:
                            return hidden_states[:, -1, :]
                    elif emb_type == 'mean':
                        return torch.mean(hidden_states[:, :, :], dim=1)

                if multimodal_input:
                    img_images, img_modalities, img_sizes = [], [], []
                    text_images, text_modalities, text_sizes = [], [], []
                    img_cnt = 0
                    for i in range(input_image_ids.shape[0]):
                        if IMAGE_TOKEN_INDEX in input_image_ids[i]:
                            img_images.append(images[img_cnt])
                            img_modalities.append(modalities[img_cnt])
                            img_sizes.append(image_sizes[img_cnt])
                            img_cnt += 1
                        else:
                            prev_img = img_cnt-1 if img_cnt-1 > 0 else img_cnt
                            img_images.append(torch.zeros_like(images[prev_img]))
                            img_modalities.append(modalities[prev_img])
                            img_sizes.append(image_sizes[prev_img])
                        if IMAGE_TOKEN_INDEX in input_text_ids[i]:
                            text_images.append(images[img_cnt])
                            text_modalities.append(modalities[img_cnt])
                            text_sizes.append(image_sizes[img_cnt])
                            img_cnt += 1
                        else:
                            prev_img = img_cnt-1 if img_cnt-1 > 0 else img_cnt
                            text_images.append(torch.zeros_like(images[prev_img]))
                            text_modalities.append(modalities[prev_img])
                            text_sizes.append(image_sizes[prev_img])

                    image_embeds = extract_emb(input_image_ids, None, image_attention_mask, None, None, img_images, img_modalities, img_sizes)
                    text_embeds = extract_emb(input_text_ids, None, text_attention_mask, None, None, text_images, text_modalities, text_sizes)
                else:
                    if images is not None:
                        image_embeds = extract_emb(input_image_ids, None, image_attention_mask, None, None, images, modalities, image_sizes)
                        if not image_only:
                            text_embeds = extract_emb(input_text_ids, None, text_attention_mask, None, None, [torch.zeros_like(i) for i in images], modalities, image_sizes)
                        else: text_embeds = None
                    else:
                        text_embeds = extract_emb(input_text_ids, None, text_attention_mask)
                        image_embeds = None

                if emb_head:
                    image_embeds = self.img_emb_head(image_embeds) if image_embeds is not None else None
                    text_embeds = self.text_emb_head(text_embeds) if text_embeds is not None else None

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) if image_embeds is not None else None
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) if text_embeds is not None else None

            if compute_embedding and language_generation:
                return outputs, text_embeds, image_embeds
            elif compute_embedding:
                return text_embeds, image_embeds
            elif language_generation:
                return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
