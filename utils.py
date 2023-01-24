

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
    
    text = text.replace('ß',"b")
    text = text.replace('à',"a")
    text = text.replace('á',"a")
    text = text.replace('ç',"c")
    text = text.replace('è',"e")
    text = text.replace('é',"e")
    text = text.replace('$',"s")
    text = text.replace("1","")
    text = text.replace("ù", "u")
    
    
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


    text = text.replace("ouw", "و")
    text = text.replace("th", "ذ")
    text = text.replace("kh", "خ")
    text = text.replace("ch", "ش")
    text = text.replace("ou", "و")
    text = text.replace("aye", "اي")
    text = text.replace("dh", "ض")
    text = text.replace("bil", "بال")
    text = text.replace("ph", "ف")
    text = text.replace("iw", "يو")
    text = text.replace("sh", "ش")
    text = text.replace("ca", "كا")
    text = text.replace("ci", "سي")
    text = text.replace("ce", "سو")
    text = text.replace("co", "كو")
    text = text.replace("ck", "ك")

    text = text.replace(" i", " ا")
    text = text.replace(" a", " ا")
    text = text.replace(" e", " ا")
    text = text.replace(" o", " ا")
    
    text = text.replace("a ", "ا ")
    text = text.replace("e ", "ا ")
    text = text.replace("i ", "ي ")
    text = text.replace("o ", "و ")
    
    text = text.replace("e", "")
    text = text.replace("a", "")
    text = text.replace("o", "")

    text = text.replace("b", "ب")
    text = text.replace("i", "")
    text = text.replace("k", "ك")
    text = text.replace("3", "ع")
    text = text.replace("5", "خ")
    text = text.replace("r", "ر")
    text = text.replace("4", "ر")
    text = text.replace("y", "ي")
    text = text.replace("s", "ص")
    text = text.replace("w", "و")
    text = text.replace("m", "م")
    text = text.replace("9", "ق")
    text = text.replace("n","ن")
    text = text.replace("d", "د")
    text = text.replace("l" ,"ل")
    text = text.replace("h", "ه")
    text = text.replace("7", "ح")
    text = text.replace("j" ,"ج")
    text = text.replace("t", "ت")
    text = text.replace("8", "غ")
    text = text.replace("2", "أ")
    text = text.replace("f", "ف")
    text = text.replace("p", "ب")
    text = text.replace("u", "و")
    text = text.replace("g", "ق")
    text = text.replace("v", "ف")
    text = text.replace("c", "س")
    text = text.replace("z", "ز")
    text = text.replace("q", "ك")
    text = text.replace("x", "اكس")
    
    
    return text.strip()