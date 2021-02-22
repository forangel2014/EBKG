"""Pre-trains an Energy-Based Knowledge Graph."""

import argparse
import os
import re
import logging
import numpy as np
import random
#import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
#import pytorch_lightning as pl
from metric import evaluate

class TripleDataSet():

    def __init__(self, data_dir, load):
        self.data_dir = data_dir
        self.load = load
        self.entity_file = os.path.join(data_dir, 'all_entities.txt')
        self.relation_file = os.path.join(data_dir, 'all_relations.txt')
        self.train_triple_file = os.path.join(data_dir, 'train_triples.txt')
        self.valid_triple_file = os.path.join(data_dir, 'valid_triples.txt')
        self.test_triple_file = os.path.join(data_dir, 'test_triples.txt')
        self.train_triple_text_file = os.path.join(data_dir, 'train_triples_text.txt')
        self.valid_triple_text_file = os.path.join(data_dir, 'valid_triples_text.txt')
        self.test_triple_text_file = os.path.join(data_dir, 'test_triples_text.txt')
        if not self.load:
            self.entity_dict = self.build_LUT(self.entity_file)
            self.relation_dict = self.build_LUT(self.relation_file)

    def build_LUT(self, file):
        lut = {}
        with open(file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = re.split('[\t\n]', line)
                lut.update({line[0]:line[1]})
        return lut

    def build_train_set(self):
        if not self.load:
            with open(self.train_triple_file, 'r', encoding='UTF-8') as f:
                all_triples = f.readlines()
                triple_num = len(all_triples)
                dataset = []
                for triple in all_triples:
                    triple = re.split('[\t\n]', triple)
                    try:
                        text_h = self.entity_dict[triple[0]]
                        text_r = self.relation_dict[triple[1]]
                        text_t = self.entity_dict[triple[2]]
                    except:
                        continue
                    dataset.append([text_h, text_r, text_t])
            with open(self.train_triple_text_file, 'w', encoding='UTF-8') as f:
                for line in dataset:
                    string = line[0] + '\t' + line[1] + '\t' + line[2] + '\n'
                    f.writelines(string)
        else:
            with open(self.train_triple_text_file, 'r', encoding='UTF-8') as f:
                all_triples = f.readlines()
                triple_num = len(all_triples)
                dataset = []
                for triple in all_triples:
                    triple = re.split('[\t\n]', triple)
                    dataset.append([triple[0], triple[1], triple[2]])
        return dataset

    def build_valid_set(self):
        if not self.load:
            with open(self.valid_triple_file, 'r', encoding='UTF-8') as f:
                all_triples = f.readlines()
                triple_num = len(all_triples)
                dataset = []
                for triple in all_triples:
                    triple = re.split('[\t \n]', triple)
                    try:
                        text_h = self.entity_dict[triple[0]]
                        text_r = self.relation_dict[triple[1]]
                        text_t = self.entity_dict[triple[2]]
                    except:
                        continue
                    dataset.append([text_h, text_r, text_t])
            with open(self.valid_triple_text_file, 'w', encoding='UTF-8') as f:
                for line in dataset:
                    string = line[0] + '\t' + line[1] + '\t' + line[2] + '\n'
                    f.writelines(string)
        else:
            with open(self.valid_triple_text_file, 'r', encoding='UTF-8') as f:
                all_triples = f.readlines()
                triple_num = len(all_triples)
                dataset = []
                for triple in all_triples:
                    triple = re.split('[\t\n]', triple)
                    dataset.append([triple[0], triple[1], triple[2]])
        return dataset

class TorchTripleDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample

    def __len__(self):
        return len(self.dataset)

class Config():
    def __init__(self,
                model_dir,
                model_load,
                vocab_size = 16296,
                train_method = 'pretrain gpt2',
                task = 'tail enetity prediction',
                n = 1,
                k = 1,
                valid_num = 10,
                lr = 1e-5,
                weight_decay = 0,
                batchsize = 16,
                epoch = 10):
        self.model_dir=model_dir
        self.model_load=model_load
        self.vocab_size=vocab_size
        self.train_method = train_method
        self.task = task
        self.n = n
        self.k = k
        self.valid_num = valid_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.batchsize = batchsize
        self.epoch = epoch

class EnergyBasedKGModel(torch.nn.Module):
    def __init__(self, config: Config):
        super(EnergyBasedKGModel, self).__init__()
        self.is_trained = False
        self.config = config

        #generator
        self.generator_name = "gpt2"
        self.generator_path = "/home/sunwangtao/EBKG/generator_file/"
        self.generator_config = GPT2Config.from_pretrained(self.generator_path)
        self.generator_tokenizer = GPT2Tokenizer.from_pretrained(self.generator_path)
        self.generator_model = GPT2LMHeadModel.from_pretrained(self.generator_path)
        #self.generator_special_tokens_dict = {'cls_token': '[SEP]'}
        #self.generator_tokenizer.add_special_tokens(self.generator_special_tokens_dict)

        #self.generator_config.save_pretrained(self.generator_path)
        #self.generator_tokenizer.save_pretrained(self.generator_path)
        #self.generator_model.save_pretrained(self.generator_path)
        #self.generator_model.resize_token_embeddings(len(self.generator_tokenizer))
        #self.generator_model.init_weights()
        
        #discriminator
        self.discriminator_name = "bert-base-uncased"
        self.discriminator_path = "/home/sunwangtao/EBKG/discriminator_file/"
        self.discriminator_config = BertConfig.from_pretrained(self.discriminator_path)
        self.discriminator_tokenizer = BertTokenizer.from_pretrained(self.discriminator_path)
        self.discriminator_model = BertModel.from_pretrained(self.discriminator_path)

        self.discriminator_special_tokens_dict = {'bos_token': '<|endoftext|>'}
        self.discriminator_tokenizer.add_special_tokens(self.discriminator_special_tokens_dict)
        self.discriminator_model.resize_token_embeddings(len(self.discriminator_tokenizer))
        self.discriminator_linear = torch.nn.Linear(self.discriminator_config.hidden_size, 1)

        #self.discriminator_config.save_pretrained(self.discriminator_path)
        #self.discriminator_tokenizer.save_pretrained(self.discriminator_path)
        #self.discriminator_model.save_pretrained(self.discriminator_path)
        
        #self.discriminator_model.init_weights()
        #initialize
        self.train_mode()

    def train_mode(self):
        self.train()
        self.generator_model.train()
        self.discriminator_model.train()

    def eval_mode(self):
        self.eval()
        self.generator_model.eval()
        self.discriminator_model.eval()

    def generate(self, l, text='<|endoftext|>'):
        generate_text = text
        logq = 0
        indexed_tokens = self.generator_tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        for i in range(l):
            repr_vec = self.get_repr_vec(self.generator_model, tokens_tensor)[-1,:]
            q_distribution = F.softmax(repr_vec)
            #generate_index = self.sample_top_k(repr_vec)
            generate_index = self.sample_prob(q_distribution)
            #generate_text = self.generator_tokenizer.decode(indexed_tokens + [generate_index])
            generate_text += self.generator_tokenizer.decode(generate_index)
            indexed_tokens += [generate_index]
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            logq += torch.log(q_distribution[generate_index])
            #print(generate_text)
            #print(tokens_tensor)
        generate_text += '<|endoftext|>'
        return generate_text, logq
    
    def generate_logq(self, input_tensor):
        logq = 0
        if (input_tensor.dim() == 2):
            input_tensor = input_tensor[0]
        for i in range(1, input_tensor.shape[0]-1):
            if (input_tensor[i].cpu().detach().numpy() == -1):
                break
            repr_vec = self.get_repr_vec(self.generator_model, input_tensor[0:i].view([i]))[-1,:]
            q_distribution = F.softmax(repr_vec)
            logq += torch.log(q_distribution[input_tensor[i]])
        return logq

    def get_repr_vec(self, model, x):
        model_output = model(x)[0]
        if (model_output.dim() == 3):
            model_output = model_output[0]
        repr_vec = model_output
        return repr_vec

    def sample_prob(self, q):
        rand = random.random()
        q = q.cpu().detach().numpy()
        for i in range(len(q)):
            rand -= q[i]
            if (rand <= 0):
                return i
        '''
        a = np.asarray(q)
        choices = np.prod(a.shape)
        index = np.random.choice(choices, size=n, p=a.ravel())
        return np.unravel_index(index, dims=a.shape)
        '''
    def sample_top_k(self, predictions, k=10):
        predicted_index = random.choice(predictions.sort(descending=True)[1][:k]).item()
        return predicted_index

    def discriminate(self, tensor):
        repr_vec = self.get_repr_vec(self.discriminator_model, tensor)
        E = self.discriminator_linear(repr_vec[0,:])
        #p_hat = torch.exp(-E)
        return E

    def forward(self, input_tensor):

        if (self.config.train_method == 'pretrain gpt2'):
            #input_tensor = self.triple2tensor(self.generator_tokenizer, triple)
            for i in range(input_tensor.shape[0]):
                logq = self.generate_logq(input_tensor[i,:])
                if i == 0:
                    loss = -logq
                else:
                    loss -= logq
            return loss

        if (self.config.train_method == 'nce'):
            loss_data = torch.tensor(0)
            loss_noise = torch.tensor(0)
            for i in range(input_tensor.shape[0]):
                #input_tensor = self.triple2tensor(self.generator_tokenizer, input_triple)
                logq = self.generate_logq(input_tensor[i,:])
                #input_tensor = self.triple2tensor(self.discriminator_tokenizer, input_triple)
                E = self.discriminate(input_tensor[i,:])
                #loss = -log(n*p_hat/(n*p_hat+k*q)) = -log(1/(1+k/n*q/p_hat)) = -log(1/(1+exp(log(k/n)+log(q)-log(p_hat))))
                loss_data += -torch.log(torch.sigmoid(torch.log(n)-torch.log(k)-E-logq))
            for i in range(int(input_tensor.shape[0]/n*k)):
                l = self.sample_prob(self.l_distribution)
                generate_seq, logq = self.generate(l)
                generate_tensor = self.discriminator_tokenizer.encode(generate_seq, return_tensors="pt").cuda()
                E = self.discriminate(generate_tensor)
                #loss = -log(k*q/(n*p_hat+k*q)) = -log(1/(1+n/k*p_hat/q)) = -log(1/(1+exp(log(n/k)+log(p_hat)-log(q))))
                loss_noise += -torch.log(torch.sigmoid(torch.log(k)-torch.log(n)+logq+E))
            loss = loss_data + loss_noise
            return loss

    def triple2tensor(self, tokenizer, triple):
        text = '<|endoftext|> ' + triple[0] + ' <|endoftext|> ' + triple[1] + ' <|endoftext|> ' + triple[2] + ' <|endoftext|>'
        seq = tokenizer.encode(text, return_tensors="pt").cuda()
        return seq

    def stat_length(self, trainset):
        self.l_distribution = torch.zeros(200)
        for triple in trainset:
            seq = self.triple2tensor(self.generator_tokenizer, triple)
            l = seq.shape[1]
            self.l_distribution[l-2] += 1
        self.l_distribution /= torch.sum(self.l_distribution)

def train_and_eval(model, tripleDataSet):
    trainset = tripleDataSet.build_train_set()
    model.module.train_mode()
    optimizer = optim.Adam(model.module.parameters(), lr=model.module.config.lr, weight_decay=model.module.config.weight_decay)
    train_num = len(trainset)
    model.module.stat_length(trainset)

    for e in range(model.module.config.epoch):
        random.shuffle(trainset)
        for b in range(train_num // model.module.config.batchsize):
            optimizer.zero_grad()
            triple = trainset[b*model.module.config.batchsize:(b+1)*model.module.config.batchsize]
            max_len = 0
            for i in range(model.module.config.batchsize):
                triple[i] = model.module.triple2tensor(model.module.generator_tokenizer, triple[i])
                if (len(triple[i][0]) > max_len):
                    max_len = len(triple[i][0])
            
            input_tensor = torch.cat((triple[0], (-torch.ones([1, max_len-len(triple[0][0])])).type_as(triple[0]).cuda(0)), 1)
            for i in range(1, model.module.config.batchsize):
                tensor_pad = torch.cat((triple[i], (-torch.ones([1, max_len-len(triple[i][0])])).type_as(triple[0]).cuda(0)), 1)
                input_tensor = torch.cat((input_tensor, tensor_pad), 0)
            loss = model(input_tensor).sum()

            loss.backward()
            optimizer.step()
            loss_num = loss.cpu().detach().numpy()
            print("epoch "+str(e)+" step "+str(b)+" loss = "+str(loss_num))
            logging.debug(loss_num)
            
            if (b % 100 == 0):
                torch.save(model.module, os.path.join(model.module.config.model_dir,'checkpoint.pt'))
                if (model.module.config.train_method == 'pretrain gpt2'):
                    for _ in range(10):
                        l = model.module.sample_prob(model.module.l_distribution)
                        generate_text, logq1 = model.module.generate(l)
                        tensor = model.module.generator_tokenizer.encode(generate_text, return_tensors="pt").cuda()
                        logq2 = model.module.generate_logq(tensor)
                        print(generate_text)
                        print(logq1.cpu().detach().numpy(),logq2.cpu().detach().numpy())
                if (model.module.config.train_method == 'nce'):
                    model.valid(tripleDataSet)

def valid(model, tripleDataSet):
    #model.module.eval_mode()
    validset = tripleDataSet.build_valid_set()
    random.shuffle(validset)
    pair = []
    if self.config.task == 'tail enetity prediction':
        entity_dict = tripleDataSet.build_LUT(tripleDataSet.entity_file)
        for i in range(self.config.valid_num):
            triple = validset[i]
            t_true = triple[2]
            l = self.sample_prob(self.l_distribution)
            text = '<|endoftext|> ' + triple[0] + ' <|endoftext|> ' + triple[1] + ' <|endoftext|> '
            seq = self.generator_tokenizer.encode(text, return_tensors="pt").cuda()
            l -= seq.shape[1]
            tail_text = self.generate(l, text=text)[0][len(text):-13]
            print(tail_text)
            #pair.append([t_true,t_pred])
        #result = evaluate(pair)
        #print(result)
    if self.config.task == 'relation prediction':
        relation_dict = tripleDataSet.build_LUT(tripleDataSet.relation_file)
        for i in range(self.config.valid_num):
            input_triple = validset[i]
            r_true = input_triple[1]
            p_pred = []
            r_pred = []
            for r in relation_dict.items():
                r = r[1]
                input_triple[1] = r
                input_tensor = self.triple2tensor(self.discriminator_tokenizer, input_triple)
                p_hat = self.discriminate(input_tensor).cpu().detach().numpy()[0]
                r_pred.append(r)
                p_pred.append(p_hat)
            r_pred = [r_pred for _,r_pred in sorted(zip(p_pred,r_pred),reverse=True)]
            pair.append([r_true,r_pred])
        result = evaluate(pair)
        print(result)
    #model.module.train_mode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    tripleDataSet = TripleDataSet(data_dir=args.data_dir, load=True)

    config = Config(model_dir=args.model_dir, model_load=True)

    if config.model_load:
        ebkgm = torch.load(os.path.join(config.model_dir,'checkpoint.pt'))
        filemode = 'a'
    else:
        ebkgm = EnergyBasedKGModel(config=config)
        filemode = 'w'

    logging.basicConfig(level=logging.DEBUG,
                        filename='EBKG.log',
                        filemode=filemode,
                        format='')

    if torch.cuda.is_available():
        ebkgm.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ebkgm = torch.nn.DataParallel(ebkgm)

    ebkgm = ebkgm.cuda(0)
    train_and_eval(ebkgm, tripleDataSet)
    torch.save(ebkgm, os.path.join(args.model_dir,'checkpoint.pt'))


    pass

if __name__ == "__main__":
    main()