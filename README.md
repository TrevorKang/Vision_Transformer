# Vision Transformer

ViT with self attention and CVit with cross attention

  # trainer.py
  excute the basic training and testing process

  use the early_stopping and data augmentation with torch.transform()
  
  contains three models:
  
    ResNet-18 as pretrained model
    
    ViT as Vision Transformer
    
    CrossViT
  
  # my_models.py
  implementing the ViT and CViT
  
  ViT uses the self attention mechanism and the multi-head attention

    Attention(Q,K,V) = softmax(Q K^T/sqrt(d_k)) V
    Q is the query matrix, K is the key matrix, V is the value matrix
    d_k is the dimension of the key matrix
    
  
  CrossViT with cross attention considered the class token and patch token from both small and big tokens 
  It can fuse the global and local information better than vanilla ViT.

  The general structure of CrossViT is as follows:
    
    ImageEmbedder() x2 for L and S

    Multi-scale Encoder():
    {
      Transformer for L
      Transformer for S
      Cross Transformer for L and S
      {
          ProjIn
          Attention
          ProjOut
      }
    }
      
    Classifier() as MLP head