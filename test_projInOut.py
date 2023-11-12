import torch

from test_cvit import ProjectInOut
from test_cvit import Attention


sm_cls = torch.rand(16, 1, 64)
sm_patch_tokens = torch.rand(16, 16, 64)

lg_cls = torch.randn(16, 1, 128)
lg_patch_tokens = torch.randn(16, 4, 128)

sm_dim = 64
lg_dim = 128
heads = 8
dim_head = 64
dropout = 0.1

small_attention_layer = Attention(dim=sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)
crs_lg = ProjectInOut(dim_outer=lg_dim, dim_inner=sm_dim, fn=small_attention_layer)
sm2lg = crs_lg(lg_cls, context=sm_patch_tokens, kv_include_self=True)
print(sm2lg.shape)
