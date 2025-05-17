# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from PIL import Image
from typing import Optional, Union, Tuple, Dict

def adain_op(hidden_states):
    style_hidden_states, source_hidden_states = hidden_states.chunk(2, dim=0)

    style_mean = style_hidden_states.mean(dim=1, keepdim=True)
    style_std = style_hidden_states.std(dim=1, keepdim=True)

    source_mean = source_hidden_states.mean(dim=1, keepdim=True)
    source_std = source_hidden_states.std(dim=1, keepdim=True)

    source_hidden_states = style_std * (source_hidden_states - source_mean) / source_std + style_mean

    hidden_states = torch.cat([style_hidden_states, source_hidden_states], dim=0)

    return hidden_states

def save_images(images,dest, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    pil_img = Image.fromarray(images[-1])
    pil_img.save(dest)

def save_image(images,dest, num_rows=1, offset_ratio=0.02):
    print(images.shape)
    pil_img = Image.fromarray(images[0])
    pil_img.save(dest)

def register_attention_control(model, controller):
    class AttnProcessor():
        def __init__(self, place_in_unet):
            self.place_in_unet = place_in_unet

            # print("place in unet:", place_in_unet)
        def __call__(self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            ):
            # The `Attention` class can call different attention processors / attention functions

            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            h = attn.heads
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            cur_att_layer = controller.cur_att_layer
            if not is_cross and cur_att_layer in [15, 16]:
                hidden_states = adain_op(hidden_states) # adain

            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)

            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)

            attention_probs = attn.get_attention_scores(q, k, attention_mask) # bh, n, n
            hidden_states = torch.bmm(attention_probs, v)
            
            if not is_cross:

                mixed_features = controller(q, k, v, attn.heads, attn)

                if mixed_features != None and hidden_states.shape[0] // attn.heads == 2:

                    hidden_states[attn.heads:] = mixed_features

            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj   
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        def between_steps(self):
            return

    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):
            # print(m.__class__.__name__)
            if m.__class__.__name__ == "Attention":
                count += 1
                m.processor = AttnProcessor(place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count
