import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # so they dont become too small compared to positional encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        #we need matrix of seq_len to d_model size coz max length of sentence is seq_len
        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #apply the formulas
        #create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #unsqueeze add dim-1  (so it becomes a column vector)
        div_term = torch.exp(torch.arange(0, d_model,2).float()* (-math.log(10000.0)/d_model))
        
        #sin only for even position and cosine for even position
        #Apply the sin to even pos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len , d_model)
        self.register_buffer('pe',pe)
    
    def forward(self, x ):
         x= x + (self.pe[:, :x.shape[1], : ]).requires_grad_(False) #take all batches, first 'n' positions, take all embedding dimensions 
         return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # we need eps coz if sigma becomes very big gpu problem when divison so eps added
        #alpha multiplicative and bias added
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True )
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x- mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model: int,d_ff:int,dropout: float) -> None:
        super().__init__()
        # performs matrix multiplication and add bias
        #expands then back to original linear 1 and 2
        self.linear_1 = nn.Linear(d_model,d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
    def forward(self,x):
        #(Batch, seq_len , d_model) --> (Batch, seq_len , d_ff) --> (Batch, seq_len , d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

"""MULTI HEAD ATTENTION - QUERY, KEY, VALUES
 Q,K,V EXACTLY SAME COZ ITS ENCODER
 MULTIPLY BY MATRIX Wq, Wk, Wv results in new matrix 
 Q' K' V' (seq,d_model) which is then split into
 H matrices (h is number of head) and we split along the embedding dimension
 not sequence dimension and apply attention to these h matrices which will result in
 smaller matrices and then concat h1...hn and finally multiply by
 Wo to get the MULTI HEAD ATTENTION OUTPUT (maintains same dimension)
"""
#REFER IMG_1 ALONG WITH THIS

class MultiHeadAttentionBlock(nn.Module):
    def __init__ (self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model //h

        self.w_q = nn.Linear(d_model,d_model) #wq matrix
        self.w_k = nn.Linear(d_model,d_model) #wk matrix
        self.w_v = nn.Linear(d_model,d_model) #wv matrix

        self.w_o = nn.Linear(d_model,d_model) #wo matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        #(Batch,h,seq_len,d_k) --> (Batch,h,seq_len,seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # Output shape: (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0 , -1e9) #if Mask == 0 replace by -1e9
        attention_score = attention_score.softmax(dim = -1) #(Batch, h,seq_len,seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @value),attention_score # last attention_score is for visualization 
       
    """ In attention we does softmax and multiply b V we get the head matrix dv
        but before we multiply by v (only softmax) we get each word by each word matrix.
        If we dont want some words to interact with other we replace their
        value/attention score with something that is very small before softmax,
        so after softmax they become 0. This is what mask does."""
    
    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (Batch, seq_len , d_model) -->[multiplied by (d_model ,d_model)] -->  (Batch, seq_len , d_model)
        key = self.w_k(k) # (Batch, seq_len , d_model) -->[multiplied by (d_model ,d_model)] -->  (Batch, seq_len , d_model)
        value = self.w_v(v) # (Batch, seq_len , d_model) -->[multiplied by (d_model ,d_model)] -->  (Batch, seq_len , d_model)

        # (Batch, seq_len , d_model) -->(Batch, seq_len ,h, d_k) -->  (Batch, h, seq_len , d_k)
        
        """query.shape = (batch_size, seq_len, d_model)
            self.h = number of heads
            self.d_k = d_model // self.h
            view is like numpy reshape change tensor without changing data order"""
        
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose (1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value =value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #finally concat and then multiply by wo
        # (Batch,h,seq_len,d_l) --> (Batch,seq_len,h_d_k) --> (Batch,seq_len,d_k)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) #coz d_k d_model//h , after transpose memory is not contigous so we do contigous
        
        #(Batch,seq_len,d_model) --> (Batch,seq_model.d_model)
        return self.w_o(x)
    
#the skip layer - residual connection
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #First normalization then sublayer in paper it is opposite  
    """Better gradient flow ,More stable training: normalization before sublayer avoids exploding/vanishing gradients, 
    Post-LayerNorm can sometimes be unstable for very deep networks (>12 layers)"""

class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) #one for feed forward and other for selfattention

    """src_mask is the mask for the input of the encoder
        coz we want to hide interaction of padding word with other words"""
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x :self.self_attention_block(x,x,x,src_mask))#here x,x,x are query , key ,value
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    
#coz encoder is made up of many encoder blocks we can have upto 'n' of them

class Encoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    # here each layer takes the o/p of previous layer and final normalization
    def forward(self,x,mask):
        for layer in self.layers:
            x= layer(x,mask)
        return self.norm(x)

# --END OF ENCODER--

class DecoderBlock(nn.Module):
    """In the masked multi head attention there is self attention coz, same is used thrice
      and for multi head attention the key and value is coming from encoder part and query is from decoder part
      """
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # here 3 residual connection

    #why we need src_mask and tgt_mask coz here we translate language english to other 
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))#query,key,value and target mask coz decoder
        x = self.residual_connections[1](x,lambda x : self.cross_attention_block(x,encoder_output,encoder_output,src_mask)) #croz attention so query same other from encoder
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x 
     

#n times block
class Decoder (nn.Module):
    def __init__(self, layers :nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask , tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

# --END OF DECODER--
"""The decoder output is (batch_size,seq_len,_d_model) we need to convert it to
     (batch_size,seq_len,vocab_size) thats why we use linear at end"""

class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int , vocab_size:int) ->None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        #(batch_size,seq_len,_d_model) --> (batch_size,seq_len,vocab_size) 
        #softmax - logits to probability
        return torch.log_softmax(self.proj(x),dim = -1)

#TRANSFORMER
"""src_tokens → src_embed → src_pos → encoder → encoder_output
tgt_tokens → tgt_embed → tgt_pos → decoder(encoder_output) → decoder_output
decoder_output → projection → logits → softmax → next token"""

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder :Decoder,src_embed: InputEmbeddings,tgt_embed: InputEmbeddings,src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    #3 methods 1 to encode ,1 to decode , 1 to project
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask) # basically forward method of decoder
    
    def project(self,x):
        return self.projection_layer(x)
    
#given all yper parameter build the transformer adn inital values
#here language translation
#if token need for src lang is much lower or greate, then dont keep same, keep different
def build_transformer(src_vocab_size :int, tgt_vocab_size :int, src_seq_len : int, tgt_seq_len:int, d_model: int =512 , N:int =6, h:int =8,dropout:float = 0.1,d_ff = 2048) -> Transformer:
    #the values above are all according to the paper
    #create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #create position encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)

    #create encoder blocks 'n'
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    #create decoder blocks 'n'
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    #create encode and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))# all n blocks
    decoder = Decoder(nn.ModuleList(decoder_blocks))# all n blocks

    #create the projection layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    #create transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    #Initialize the parameters
    # Initialize weights so activations and gradients don’t explode or vanish
    #W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    for p in transformer.parameters():
        if p.dim() > 1: # apply only to weight matrices (not biases or LayerNorm vectors)
            nn.init.xavier_uniform_(p)
    return transformer
