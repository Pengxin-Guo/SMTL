import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel # https://huggingface.co/docs/transformers/model_doc/bert
from transformers.models.bert.modeling_bert import BertLayer

class BaseModel(nn.Module):
    def __init__(self, task_num):
        super(BaseModel, self).__init__()
        
        self.task_num = task_num
        self.rep_detach = False
        self.loss_weight_init = None
        
        if self.rep_detach:
            self.rep = [0]*self.task_num
            self.rep_i = [0]*self.task_num
        if isinstance(self.loss_weight_init, float):
            self.loss_scale = nn.Parameter(torch.FloatTensor([self.loss_weight_init]*self.task_num))
        
    def forward(self):
        pass


def compute_loss(logits, data, label_num, task_type=None, task=None):
    if task_type == 'TC' or task in ['panx', 'udpos']:
        loss_fct = nn.CrossEntropyLoss()
        if data['attention_mask'] is not None:
            active_loss = data['attention_mask'].view(-1) == 1
            active_logits = logits.view(-1, label_num)
            active_labels = torch.where(
                active_loss, data['labels'].view(-1), torch.tensor(loss_fct.ignore_index).type_as(data['labels'])
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, label_num), data['labels'].view(-1))
        return (loss, logits)
    elif task_type == 'SC' or task in ['xnli', 'pawsx']:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, label_num), data['labels'].view(-1))
        return (loss, logits)
    else:
        raise('no support')

class mBert(BaseModel):
    def __init__(self, label_num, task_num, task_type='TC'):
        super(mBert, self).__init__(task_num=task_num)
        # task_type: TC(NER, POS), SC(XNIL, PAWSX)
        self.task_num = task_num
        self.label_num = label_num
        self.task_type = task_type
        
        add_pooling_layer = True if task_type == 'SC' else False
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer)
        
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1, inplace=False) for _ in range(self.task_num)])
        self.fc = nn.ModuleList([nn.Linear(768, self.label_num) for _ in range(self.task_num)])
         
    def forward(self, data, task_index):
        outputs = self.bert(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep = outputs[1] if self.task_type=='SC' else outputs[0]
        if self.rep_detach:
            self.rep[task_index] = rep
            self.rep_i[task_index] = rep.detach().clone()
            self.rep_i[task_index].requires_grad = True
            rep = self.rep_i[task_index]
        
        if self.task_type == 'TC':
            sequence_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](sequence_output)
            loss = compute_loss(logits=logits, task_type='TC', data=data, label_num=self.label_num)
            return loss
        elif self.task_type == 'SC':
            pooled_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](pooled_output)
            loss = compute_loss(logits=logits, task_type='SC', data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
    
    def predict(self, data, task_index):
        return self.forward(data, task_index)


class STL(BaseModel):
    def __init__(self, label_num, task_num, task_type='TC'):
        super().__init__(task_num=task_num)
        # task_type: TC(NER, POS), SC(XNIL, PAWSX)
        self.task_num = task_num
        self.label_num = label_num
        self.task_type = task_type
        
        add_pooling_layer = True if task_type == 'SC' else False
        
        # self.embedding = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer).embeddings
        self.berts = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer)

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.fc = nn.Linear(768, self.label_num)
        
    def forward(self, data, task_index):
        outputs = self.berts(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep = outputs[1] if self.task_type=='SC' else outputs[0]
        if self.rep_detach:
            self.rep[task_index] = rep
            self.rep_i[task_index] = rep.detach().clone()
            self.rep_i[task_index].requires_grad = True
            rep = self.rep_i[task_index]
        
        if self.task_type == 'TC':
            sequence_output = self.dropout(rep)
            logits = self.fc(sequence_output)
            loss = compute_loss(logits=logits, task_type='TC', data=data, label_num=self.label_num)
            return loss
        elif self.task_type == 'SC':
            pooled_output = self.dropout(rep)
            logits = self.fc(pooled_output)
            loss = compute_loss(logits=logits, task_type='SC', data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
    
    def predict(self, data, task_index):
        return self.forward(data, task_index)


class SMTL_mBert(BaseModel):
    def __init__(self, label_num, task_num, task_type='TC', version='v1'):
        super(SMTL_mBert, self).__init__(task_num=task_num)
        # task_type: TC(NER, POS), SC(XNIL, PAWSX)
        self.task_num = task_num
        self.label_num = label_num
        self.task_type = task_type
        self.version = version
        
        add_pooling_layer = True if task_type == 'SC' else False
        
        # shared encoder
        self.bert_s = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer)
        # task-specific encoder
        self.bert_t = nn.ModuleList([BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer) for _ in range(self.task_num)])
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(self.task_num, 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(self.task_num))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
        
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1, inplace=False) for _ in range(self.task_num)])
        self.fc = nn.ModuleList([nn.Linear(768, self.label_num) for _ in range(self.task_num)])
        
    def forward(self, data, task_index):
        # shared encoder output
        outputs_s = self.bert_s(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_s = outputs_s[1] if self.task_type=='SC' else outputs_s[0]
        # task-specific encoder output
        outputs_t = self.bert_t[task_index](input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_t = outputs_t[1] if self.task_type=='SC' else outputs_t[0]

        # combine shared encoder output and task-specific encoder output, obtain final hidden feature
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # SMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # SMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for SMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            temp_alpha = torch.stack([1-temp, temp])
            temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
        else:
            print("No correct version parameter!")
            exit()

        rep = temp_alpha[0] * rep_s + temp_alpha[1] * rep_t

        if self.task_type == 'TC':
            sequence_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](sequence_output)
            loss = compute_loss(logits=logits, task_type='TC', data=data, label_num=self.label_num)
            return loss
        elif self.task_type == 'SC':
            pooled_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](pooled_output)
            loss = compute_loss(logits=logits, task_type='SC', data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
    
    def predict(self, data, task_index):
        # shared encoder output
        outputs_s = self.bert_s(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_s = outputs_s[1] if self.task_type=='SC' else outputs_s[0]
        # task-specific encoder output
        outputs_t = self.bert_t[task_index](input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_t = outputs_t[1] if self.task_type=='SC' else outputs_t[0]

        # combine shared encoder output and task-specific encoder output, obtain final hidden feature
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # SMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # SMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for SMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            if temp >= 0.5:
                temp_alpha = [0, 1]
            else:
                temp_alpha = [1, 0]
        else:
            print("No correct version parameter!")
            exit()

        rep = temp_alpha[0] * rep_s + temp_alpha[1] * rep_t

        if self.task_type == 'TC':
            sequence_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](sequence_output)
            loss = compute_loss(logits=logits,
                               task_type='TC',
                               data=data, label_num=self.label_num)
            return loss
            
        elif self.task_type == 'SC':
            pooled_output = self.dropout[task_index](rep)
            logits = self.fc[task_index](pooled_output)
            loss = compute_loss(logits=logits,
                               task_type='SC',
                               data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
    
    def get_adaptative_parameter(self):
        return self.alpha


class SMTL_new_mBert(BaseModel):
    def __init__(self, label_num, task_num, task_type='TC', version='v1'):
        super(SMTL_new_mBert, self).__init__(task_num=task_num)
        # task_type: TC(NER, POS), SC(XNIL, PAWSX)
        self.task_num = task_num
        self.label_num = label_num
        self.task_type = task_type
        self.version = version
        
        add_pooling_layer = True if task_type == 'SC' else False
        
        # shared encoder
        self.bert_s = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer)
        # task-specific encoder
        self.bert_t = nn.ModuleList([BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=add_pooling_layer) for _ in range(self.task_num)])
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(self.task_num, 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(self.task_num))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
        
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1, inplace=False) for _ in range(self.task_num)])
        # task-specific decoder for shared encoder
        self.fc_s = nn.ModuleList([nn.Linear(768, self.label_num) for _ in range(self.task_num)])
        # task-specific decoder for task-specific encoder
        self.fc_t = nn.ModuleList([nn.Linear(768, self.label_num) for _ in range(self.task_num)])
        
    def forward(self, data, task_index):
        # shared encoder output
        outputs_s = self.bert_s(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_s = outputs_s[1] if self.task_type=='SC' else outputs_s[0]
        # task-specific encoder output
        outputs_t = self.bert_t[task_index](input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_t = outputs_t[1] if self.task_type=='SC' else outputs_t[0]
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # SMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # SMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for SMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            temp_alpha = torch.stack([1-temp, temp])
            temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
        else:
            print("No correct version parameter!")
            exit()
            
        if self.task_type == 'TC':
            sequence_output_s = self.dropout[task_index](rep_s)
            sequence_output_t = self.dropout[task_index](rep_t)
            logits_s = self.fc_s[task_index](sequence_output_s)
            logits_t = self.fc_t[task_index](sequence_output_t)
            logits = temp_alpha[0] * logits_s + temp_alpha[1] * logits_t
            loss = compute_loss(logits=logits, task_type='TC', data=data, label_num=self.label_num)
            return loss
        elif self.task_type == 'SC':
            pooled_output_s = self.dropout[task_index](rep_s)
            pooled_output_t = self.dropout[task_index](rep_t)
            logits_s = self.fc_s[task_index](pooled_output_s)
            logits_t = self.fc_t[task_index](pooled_output_t)
            logits = temp_alpha[0] * logits_s + temp_alpha[1] * logits_t
            loss = compute_loss(logits=logits, task_type='SC', data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')


    
    def predict(self, data, task_index):
        # shared encoder output
        outputs_s = self.bert_s(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_s = outputs_s[1] if self.task_type=='SC' else outputs_s[0]
        # task-specific encoder output
        outputs_t = self.bert_t[task_index](input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids'])
        rep_t = outputs_t[1] if self.task_type=='SC' else outputs_t[0]
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # SMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # SMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for SMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            if temp >= 0.5:
                temp_alpha = [0, 1]
            else:
                temp_alpha = [1, 0]
        else:
            print("No correct version parameter!")
            exit()
            
        if self.task_type == 'TC':
            sequence_output_s = self.dropout[task_index](rep_s)
            sequence_output_t = self.dropout[task_index](rep_t)
            logits_s = self.fc_s[task_index](sequence_output_s)
            logits_t = self.fc_t[task_index](sequence_output_t)
            logits = temp_alpha[0] * logits_s + temp_alpha[1] * logits_t
            loss = compute_loss(logits=logits, task_type='TC', data=data, label_num=self.label_num)
            return loss
        elif self.task_type == 'SC':
            pooled_output_s = self.dropout[task_index](rep_s)
            pooled_output_t = self.dropout[task_index](rep_t)
            logits_s = self.fc_s[task_index](pooled_output_s)
            logits_t = self.fc_t[task_index](pooled_output_t)
            logits = temp_alpha[0] * logits_s + temp_alpha[1] * logits_t
            loss = compute_loss(logits=logits, task_type='SC', data=data, label_num=self.label_num)
            return loss
        else:
            raise('no support')
    
    def get_adaptative_parameter(self):
        return self.alpha