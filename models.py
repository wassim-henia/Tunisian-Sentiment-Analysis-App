from simpletransformers.classification import ClassificationModel
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
import streamlit as st
import torch.nn as nn

from utils import BertDataset, preprocessing_for_bert, removeDuplicates, convert
from scipy.special import softmax

@st.cache(allow_output_mutation=True)
class SimpleTransformer():

    def __init__(self, model_type, model_name, n_epochs = 2, train_batch_size = 100, eval_batch_size = 64, seq_len = 120, lr = 2e-5):
        self.model = ClassificationModel(model_type, model_name,num_labels=3,use_cuda=False, args={'train_batch_size':train_batch_size,
                                                                            "eval_batch_size": eval_batch_size,
                                                                            'reprocess_input_data': True,
                                                                            'overwrite_output_dir': True,
                                                                            'fp16': False,
                                                                            'do_lower_case': False,
                                                                            'num_train_epochs': n_epochs,
                                                                            'max_seq_length': seq_len,
                                                                            'manual_seed': 2,
                                                                            "learning_rate":lr,
                                                                            "save_eval_checkpoints": False,
                                                                            "save_model_every_epoch": False,
                                                                            "use_multiprocessing" :False,
                                                                            "use_multiprocessed_decoding": False,
                                                                            "use_multiprocessing_for_evaluation":False,
                                                                            'no_cache': True,
                                                                            "cache_dir": "ppp_gl3/"})
    
    def process(self, text):
        
        return removeDuplicates(list(text.rstrip()))
    
    def predict(self, text):

        pred = self.model.predict([text])
        simple_probs = softmax(pred[1],axis=1)

        return simple_probs



@st.cache(allow_output_mutation=True)
class Bert():

    def __init__(self, model_name,path):
        
        if torch.cuda.is_available():   

            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:

            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.model_name = model_name
        self.model = torch.load(path, map_location=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name ,do_lower_case=True)
        self.pad = self.tokenizer.pad_token_id
    
    def data_collator(self, data):
    
        sentence, mask, label, length = zip(*data)
        
        tensor_dim = max(length)
        
        out_sentence = torch.full((len(sentence), tensor_dim), dtype=torch.long, fill_value=self.pad)
        out_mask = torch.zeros(len(sentence), tensor_dim, dtype=torch.long)

        for i in range(len(sentence)):
            
            out_sentence[i][:len(sentence[i])] = torch.Tensor(sentence[i])
            out_mask[i][:len(mask[i])] = torch.Tensor(mask[i])
        
        if label[0] != None:

            return (out_sentence, out_mask, torch.Tensor(label).long())

        else:

            return (out_sentence, out_mask)

    def process(self, text):

        if self.model_name == "moha/mbert_ar_c19":

            text = convert(text)
        inputs, masks = preprocessing_for_bert([text], self.tokenizer)
        dataset = BertDataset(inputs, masks)
        sample = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sample, batch_size=128, collate_fn=self.data_collator)
   
        return dataloader

    def predict(self, test_dataloader):

        self.model.eval()

        all_logits = []
        for batch in tqdm(test_dataloader):
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        all_logits = torch.cat(all_logits, dim=0)

        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        return probs

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 200, 3
#768,100,3
        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained("bert-base")
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(H, D_out),      
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        return logits