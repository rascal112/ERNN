import torch
import torch.nn as nn
from einops import rearrange

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        in_equ,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        equ_layer = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                equ_layer.append(attention_equ('self', in_equ, num_heads))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                equ_layer.append(attention_equ('cross', in_equ, num_heads))
        self.layers = nn.ModuleList(layers)
        self.equ_layers = nn.ModuleList(equ_layer)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1, equ0, equ1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
                equ0, equ1 = self.equ_layers[i]([scores0, scores1], equ0, equ1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                    new_equ0, new_equ1 = self.equ_layers[i]([scores0, scores1], equ0, equ1)
                    equ0 = new_equ0
                    equ1 = new_equ1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    equ0, equ1 = self.equ_layers[i]([scores0, scores1], equ0, equ1)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, equ0, equ1, attention_scores
        else:
            return feats0, feats1, equ0, equ1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
        
class attention_equ(nn.Module):
    def __init__(self, block, in_channels, head):
        super(attention_equ, self).__init__()
        self.block = block
        self.num_heads = head
        # self.proj_q = nn.Linear(in_channels, in_channels, bias=False)
        # self.proj_k = nn.Linear(in_channels, in_channels, bias=False)
        self.proj_v = nn.Linear(in_channels, in_channels, bias=False)

        self.output = nn.Linear(in_channels, in_channels, bias=False)
        # self.output2 = nn.Linear(in_channels, in_channels, bias=False)
        # self.equ_norm = EquivariantLayerNormV2_channel(1, in_channels)

    def attention(self, equ, attention):
        # q = rearrange(self.proj_q(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        # k = rearrange(self.proj_k(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        l = equ.shape[2]
        v = rearrange(self.proj_v(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        attention = attention.unsqueeze(2)
        attention = attention.expand(-1, -1, l, -1, -1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h l n c -> b n l (h c)', h=self.num_heads)
        out = self.output(out)
        return out

    def forward(self, attention, equ1, equ2):
        if self.block == 'self':
            equ_new1 = self.attention(equ1, attention[0]) + equ1
            equ_new2 = self.attention(equ2, attention[1]) + equ2
        else:
            equ_new1 = self.attention(equ2, attention[0]) + equ1
            equ_new2 = self.attention(equ1, attention[1]) + equ2
        # equ_new1 = (self.equ_norm(equ_new1.squeeze(0)).unsqueeze(0))
        # equ_new2 = (self.equ_norm(equ_new2.squeeze(0)).unsqueeze(0))
        return equ_new1, equ_new2
            
