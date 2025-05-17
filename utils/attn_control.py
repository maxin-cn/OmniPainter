import torch
import abc
from einops import rearrange


def adain_op(hidden_states):
    style_hidden_states, source_hidden_states = hidden_states.chunk(2, dim=0)

    style_mean = style_hidden_states.mean(dim=1, keepdim=True)
    style_std = style_hidden_states.std(dim=1, keepdim=True)

    source_mean = source_hidden_states.mean(dim=1, keepdim=True)
    source_std = source_hidden_states.std(dim=1, keepdim=True)

    source_hidden_states = style_std * (source_hidden_states - source_mean) / source_std + style_mean

    hidden_states = torch.cat([style_hidden_states, source_hidden_states], dim=0)

    return hidden_states

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def start_att_layers(self):
        return self.start_ac_layer #if LOW_RESOURCE else 0
    @property
    def end_att_layers(self):
        return self.end_ac_layer
    
    @abc.abstractmethod
    def forward(self, q, k, v, num_heads, attn):
        raise NotImplementedError

    def attn_forward(self, q, k, v, num_heads, attn):
        if q.shape[0] // num_heads == 2: # no classfier guidance
            mixed_features = self.forward(q, k, v, num_heads, attn)
        else:
            raise ValueError("AttentionControl only supports 2 heads for now.")

        return mixed_features
    
    def __call__(self, q, k, v, num_heads, attn):

        if self.cur_att_layer >= self.start_att_layers and self.cur_att_layer < self.end_att_layers and self.cur_step < 1000: 
            mixed_features = self.attn_forward(q, k, v, num_heads, attn) 
        else:
            mixed_features = None
        
        self.cur_att_layer += 1 
        if self.cur_att_layer == self.num_att_layers // 2: #+ self.num_uncond_att_layers:
            self.cur_att_layer = 0 #self.num_uncond_att_layers
            self.cur_step += 1
            self.between_steps()

        return mixed_features

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStyle(AttentionControl):

    def __init__(self, 
                 num_steps,
                 start_ac_layer, 
                 end_ac_layer,
                 sty_guidance=0.3,
                 ):
        super(AttentionStyle, self).__init__()

        self.start_ac_layer = start_ac_layer
        self.end_ac_layer = end_ac_layer
        self.num_steps=num_steps
        self.sty_guidance = sty_guidance
    
    def forward(self, q, k, v, num_heads, attn, **kwargs):

        b, n, d = k.shape

        sq, cq = q[:num_heads], q[num_heads:]
        sk, ck = k[:num_heads], k[num_heads:]
        sv, cv = v[:num_heads], v[num_heads:]

        content_content_out_sim = self.get_batch_sim('cc', cq, ck, num_heads, attn, **kwargs)
        content_style_out_sim = self.get_batch_sim('cs', cq, sk, num_heads, attn, **kwargs)

        content_style_out_sim *= self.sty_guidance # 1.2

        content_style_content_content_out_sim = torch.cat([content_style_out_sim, content_content_out_sim], dim=2)

        content_style_content_content_out_sim = content_style_content_content_out_sim.softmax(-1)

        v_cscc_concat = torch.cat([sv, cv], dim=1) # style, content

        mixup_out = torch.einsum("h i j, h j d -> h i d", content_style_content_content_out_sim, v_cscc_concat)

        return mixup_out

    def get_batch_sim(self, type, q, k, num_heads, attn, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        # v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        # sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * attn.scale
        return sim



        

    


