
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
        # TODO

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
        Q, K, V = self.w_q(context), self.w_k(context), self.w_v(context)
        # 2. split tensor by number of heads
        q, k, v, = self.split(Q), self.split(K), self.split(V)
        # 3. do scale dot product to compute similarity/attention as h_i
        h_i, v = self.scale_dot_product(q=q, k=k, v=v)
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
        score = self.softmax(scaled) @ v
        return score, v

    def concat_h(self, h_i):
        """

        :param h_i: attention from single head [b,h,n,d]
        :return: concatenation of h_i as H [b, n, (h*d)]
        """
        b, h, n, d = h_i.size()
        H = h_i.transpose(1, 2).contiguous().view(b, n, h * d)
        return H


if __name__ == '__main__':
    tensor = torch.randn(16, 17, 64)
    attention = Attention(dim=tensor.size()[-1])
    M = attention.forward(x=tensor)
    print(M.size())
