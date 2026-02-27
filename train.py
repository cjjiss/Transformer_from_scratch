import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

import warnings
from pathlib import Path #relative path -> absolute path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #precompute the encoder output and reuse it for every token we get from decoder
    encoder_output = model.encode(source,source_mask) #encoder input and encoder mask
    #intitialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        #build mask for the decoder_input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) #here we dont need other mask coz no padding
        
        #calculate the output of the decoder
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        #get the next token
        prob = model.project(out[:,-1])
        #select the token with max probability (coz its a greedy search)
        _, next_word = torch.max(prob,dim =1)
        #it will become the input for next iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim =1)

        if next_word == eos_idx:
            break
    
    #coz we appending next token to it each iteration
    return decoder_input.squeeze(0) # remove batch dimension

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_step,num_examples=2):
    model.eval() # we are going to evalute model
    count = 0

    #size of control window (just use a default value)
    console_width = 80

    with torch.no_grad(): # disable gradient calculation
        for batch in validation_ds:
            count += 1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) ==1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            #comapre with label/target

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            #print to console
            print_msg('-'*console_width) #just so it dosent interefere with tqdm
            print_msg(f'SOURCE : {source_text}')
            print_msg(f'TARGET : {target_text}')
            print_msg(f'PREDICTED : {model_out_text}')

            if count == num_examples:
                break

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang] # pairs english and other , so take other
#build the tokenizer
def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token ='[UNK]' )) #tokenizer sees a word that it does recognize in its vocabulary replace it with 'UNK'
        tokenizer.pre_tokenizer = Whitespace() #split by whitespace
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)
        #4 special tokenz - unknown replace , padding , start of sentence and end of sentence, min_freq means a word to appear in vocab has to have frequency 2
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer =trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split = 'train')

    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    #split into train and valid  90 / 10
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_size,val_ds_size = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_size,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_size,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    """ Load each sentence from src and tgt language and convert into IDs using tokenizer
     and check length, if eg 270 choose 300 coz of SOS,EOS"""
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids)) 
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True) #process each sentence 1 by 1

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model
          
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps = 1e-9)

    #resume pgm if it crashes
    initial_epoch = 0 
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    #ignore padding
    #label smoothing - reduce overconfidence give 0.1 to others
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing = 0.1).to(device)


    for epoch in range(initial_epoch,config['num_epochs']):
        model.train() #after validation model back to training
        batch_iterator = tqdm(train_dataloader,desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:


            encoder_input = batch['encoder_input'].to(device) #(batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(batch,1,1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(batch,1,seq_len,seq_len)

            #run tensors through the transformer
            encoder_output = model.encode(encoder_input,encoder_mask) #(batch,seq_len,d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) #(batch,seq_len,d_model)
            #map it back to vocab
            proj_output = model.project(decoder_output) #(batch,seq_len,tgt_vocab_size)
            #now we have output compare it to label
            label = batch['label'].to(device) #(batch,seq_len)

            #(batch,seq_len,tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1)) #this is how cross entropy need 
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            #log on tensorboard
            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush()

            #backpropagate the loss
            loss.backward()

            #update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device, lambda msg: batch_iterator.write(msg),global_step)

        #save the model at end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)
            
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)




