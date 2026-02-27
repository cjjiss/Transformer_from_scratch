import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)-> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)#dosent matter src or tgt they both contain token
        #int64 coz vocab can be more than 32 bit
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
        #length of dataset from hugging face
        return len(self.ds)
    
    def __getitem__(self,index ):
        src_target_pair = self.ds[index] #orginal text
        src_text = src_target_pair['translation'][self.src_lang] #src text
        tgt_text = src_target_pair['translation'][self.tgt_lang] #tgt text

        #convert each text into tokens then into input ids
        #tokenizer split sentence into single words and then map each word into coresponding vocab

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # gives input ids so numbers corresponding to orginal sentence as array
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #padding token to fill until it reaches seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 coz of sos and eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 coz only sos

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError('Sentence is too long ')
        
        #add sos and eos to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
                
            ]
        )

        #Add sos to the decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype =torch.int64)
            ]
        )
        #add eos to the label 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype =torch.int64)
            ]
        )

        #checking to see its same as token length, and padding works
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        """So here we basically increase size of encoder input by padding , but we dont want to interfere
         with self attention so we need masking,
          for decoder mask 'causal mask', each word can only look at prvs words (not after it) and non padding words """
        
        """unsqueeze adds dimension 1,1,seq_len
        broadcasting - if dimensions are equal or one of them is 1 repat value without physically copying"""
        return {
            "encoder_input" : encoder_input, #(seq_len)
            "decoder_input" : decoder_input, #(seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask" :(decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1,seq_len) & (1,seq_len,seq_len)
            "label" : label, #(seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
#for example for a 2x2 matrix we want to mask out anything above the diagonal
#here mask below and flip it
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal =  1).type(torch.int)
    return mask == 0



    