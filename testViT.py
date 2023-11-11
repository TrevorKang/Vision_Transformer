import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers,
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        """

        :param dim: dimensionality of the input features, size of token embeddings
        :param heads: number of heads
        :param dim_head: dimension of a single head, length of query/key/value
        :param dropout: probability of keep
        """
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        # TODO
        self.dim = dim  # shape[-1] of input tensor
        self.heads = heads  # number of heads
        self.dim_head = dim_head
        self.scale = math.sqrt(dim_head)
        # we need softmax layer and dropout
        # TODO
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        # as well as the q linear layer
        # TODO
        self.w_q = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        # and the k/v linear layer (can be realized as one single linear layer or as two individual ones)
        self.w_k = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        self.w_v = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        # TODO
        # and the output linear layer followed by dropout
        self.w_o = nn.Linear(in_features=self.heads * dim_head, out_features=dim)

    def forward(self, x, context=None, kv_include_self=False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        # b, n, _, h = *x.shape, self.heads
        # b for batch size, n is the number of tokens and d for dim, h for number of heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        # TODO: attention
        # 1. dot product with weight matrices
        Q, K, V = self.w_q(x), self.w_k(context), self.w_v(context)
        # 2. split tensor by number of heads
        q, k, v, = self.split(Q), self.split(K), self.split(V)
        # 3. do scale dot product to compute similarity/attention as h_i
        # TODO: masking?
        h_i, v = self.scale_dot_product(q=q, k=k, v=v, mask=None)
        # 4. concat h_i
        H = self.concat_h(h_i=h_i)
        # 5. pass to linear layer
        M = self.w_o(H)
        # 6. add dropout if is needed
        out = self.dropout(M)
        return out

    def split(self, tensor):
        """
        split the long tensor into #heads sub-tensor
        :param tensor: [b, n, (h*d)]
        :return: [b, h, n, d]
        """
        b, n, d_model = tensor.size()
        d_tensor = d_model // self.heads
        tensor = tensor.view(b, n, self.heads, d_tensor).transpose(1, 2)

        return tensor

    def scale_dot_product(self, q, k, v, mask=None):
        """

        :param q: query, what a token is looking for, global information
        :param k: key, description of a query, local info
        :param v: value
        :param mask: if we use masking operation to the scaled value
        :return: attention score
        """
        b, h, n, d = k.size()
        k_T = k.transpose(2, 3)  # swap n and d
        scaled = q @ k_T / self.scale
        # TODO: if add mask
        if mask is not None:
            scaled = scaled.masked_fill(mask == 0, -1e9)
        score = self.dropout(self.softmax(scaled)) @ v
        return score, v

    def concat_h(self, h_i):
        """

        :param h_i: attention from single head [b,h,n,d]
        :return: concatenation of h_i as H [b, n, (h*d)]
        """
        b, h, n, d = h_i.size()
        H = h_i.transpose(1, 2).contiguous().view(b, n, h * d)
        return H


# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        # print(self.pos_embedding[:, :(n + 1)].size())
        # print(self.pos_embedding.size())
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size=32, patch_size=8, num_classes=10, dim=64, depth=2, heads=8, mlp_dim=128, dropout=0.1,
              emb_dropout=0.1)
    print(vit(x).shape)