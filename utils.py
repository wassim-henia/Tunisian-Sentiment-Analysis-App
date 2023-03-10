

import os
import random
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset, Sampler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def removeDuplicates(S): 
          
    n = len(S)  
      
    # We don't need to do anything for  
    # empty or single character string.  
    if (n < 2) : 
        return
          
    # j is used to store index is result  
    # string (or index of current distinct  
    # character)  
    j = 0
      
    # Traversing string  
    for i in range(n):  
          
        # If current character S[i]  
        # is different from S[j]  
        if (S[j] != S[i]): 
            j += 1
            S[j] = S[i]  
      
    # Putting string termination  
    # character.  
    j += 1
    S = S[:j] 
    return "".join(S)

def preprocessing_for_bert(data, tokenizer, max_len=256):

    input_ids = []
    attention_masks = []
    tmp = tokenizer.encode("ab")[-1]

    for sentence in data:

        encoding = tokenizer.encode(sentence)

        if len(encoding) > max_len:
            encoding = encoding[:max_len-1] + [tmp]

        in_ids = encoding
        att_mask = [1]*len(encoding)
        
        input_ids.append(in_ids)
        attention_masks.append(att_mask)

    return input_ids, attention_masks

class BertDataset(Dataset):

    def __init__(self, data, masks, label=None):
        
        self.data = data
        self.masks = masks
        
        if label != None:
            self.labels = label
        else:
            self.labels = None
        
        self.lengths = [len(i) for i in data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels !=  None:
            return (self.data[idx], self.masks[idx], self.labels[idx], self.lengths[idx])
        else:  #For validation
            return (self.data[idx], self.masks[idx], None, self.lengths[idx])

def convert(text):
    
    text = text.replace('??',"b")
    text = text.replace('??',"a")
    text = text.replace('??',"a")
    text = text.replace('??',"c")
    text = text.replace('??',"e")
    text = text.replace('??',"e")
    text = text.replace('$',"s")
    text = text.replace("1","")
    text = text.replace("??", "u")
    
    
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9 ,!?.]', '', text)

    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'([h][h][h][h])\1+', r'\1', text)
    text = re.sub(r'([a-g-i-z])\1+', r'\1', text)  #Remove repeating characters
    text = re.sub(r' [0-9]+ ', " ", text)
    text = re.sub(r'^[0-9]+ ', "", text)
    
    
    text = " " + text+ " "


    text = text.replace("ouw", "??")
    text = text.replace("th", "??")
    text = text.replace("kh", "??")
    text = text.replace("ch", "??")
    text = text.replace("ou", "??")
    text = text.replace("aye", "????")
    text = text.replace("dh", "??")
    text = text.replace("bil", "??????")
    text = text.replace("ph", "??")
    text = text.replace("iw", "????")
    text = text.replace("sh", "??")
    text = text.replace("ca", "????")
    text = text.replace("ci", "????")
    text = text.replace("ce", "????")
    text = text.replace("co", "????")
    text = text.replace("ck", "??")

    text = text.replace(" i", " ??")
    text = text.replace(" a", " ??")
    text = text.replace(" e", " ??")
    text = text.replace(" o", " ??")
    
    text = text.replace("a ", "?? ")
    text = text.replace("e ", "?? ")
    text = text.replace("i ", "?? ")
    text = text.replace("o ", "?? ")
    
    text = text.replace("e", "")
    text = text.replace("a", "")
    text = text.replace("o", "")

    text = text.replace("b", "??")
    text = text.replace("i", "")
    text = text.replace("k", "??")
    text = text.replace("3", "??")
    text = text.replace("5", "??")
    text = text.replace("r", "??")
    text = text.replace("4", "??")
    text = text.replace("y", "??")
    text = text.replace("s", "??")
    text = text.replace("w", "??")
    text = text.replace("m", "??")
    text = text.replace("9", "??")
    text = text.replace("n","??")
    text = text.replace("d", "??")
    text = text.replace("l" ,"??")
    text = text.replace("h", "??")
    text = text.replace("7", "??")
    text = text.replace("j" ,"??")
    text = text.replace("t", "??")
    text = text.replace("8", "??")
    text = text.replace("2", "??")
    text = text.replace("f", "??")
    text = text.replace("p", "??")
    text = text.replace("u", "??")
    text = text.replace("g", "??")
    text = text.replace("v", "??")
    text = text.replace("c", "??")
    text = text.replace("z", "??")
    text = text.replace("q", "??")
    text = text.replace("x", "??????")
    
    
    return text.strip()