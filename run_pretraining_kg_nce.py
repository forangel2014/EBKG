"""Pre-trains an Energy-Based Knowledge Graph."""

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import re
import logging
import numpy as np
import random
import torch
from torch import optim
import torch.nn.functional as F
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer
from metric import evaluate

class Config():
    def __init__(self,
                device_ids, 
                vocab_size=16296,
                #attention_probs_dropout_prob=0.1,
                #gradient_checkpointing=False,
                #hidden_act="gelu",
                #hidden_dropout_prob=0.1,
                #hidden_size= 768,
                #initializer_range=0.02,
                #intermediate_size=3072,
                #layer_norm_eps=1e-12,
                #max_position_embeddings=512,
                #model_type="bert",
                #num_attention_heads=12,
                #num_hidden_layers=12,
                #pad_token_id=0,
                #type_vocab_size=2,
                #vocab_size=30522,
                train_method='nce',
                n = 3,
                k = 7,
                valid_num = 10,
                lr = 1e-6,
                weight_decay = 0,
                batchsize = 10,
                epoch = 10):
        self.device_ids = device_ids
        self.vocab_size=vocab_size
        #self.bert_config = bert_config
        self.train_method = train_method
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
        self.main_device = torch.device("cuda:"+str(self.config.device_ids[0]))
        #tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #generator
        self.generator_name = "bert-base-uncased"#"albert-base-v1"#
        self.generator_config = BertConfig.from_pretrained(self.generator_name)
        self.generator_model = BertModel.from_pretrained(self.generator_name)
        self.generator_linear = torch.nn.Linear([1, 3, self.generator_config.hidden_size], self.config.vocab_size)
        #discriminator
        self.discriminator_name = "bert-base-uncased"#"albert-base-v1"#
        self.discriminator_config = BertConfig.from_pretrained(self.discriminator_name)
        self.discriminator_model = BertModel.from_pretrained(self.discriminator_name)
        self.discriminator_linear = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
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

    def generate(self):
        sentence = '[CLS]'
        sep_num = 0
        sample_token = None
        q = 1
        for i in range(self.generator_config.max_positional_embeddings):
            if (sample_token == '[SEP]' and sep_num == 3):
                break
            encoded_id = self.tokenizer.encode(sentence)
            repr_vec = self.get_repr_vec(self.generator, encoded_id)[i,:]
            q_distribution = F.softmax(self.generator_linear(repr_vec))
            id_sample = self.sample(q_distribution)
            q *= q_distribution(id_sample)
            sample_token = self.tokenizer.convert_ids_to_tokens(id_sample)
            if (sample_token == '[SEP]'):
                sep_num += 1
            sentence += ' ' + sample_token

        self.tokenizer.convert_ids_to_tokens(sentence)
        return sentence, q
    
    def generate(self, input_seq):
        self.tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
        return q

    def get_repr_vec(self, model, x):
        bert_output = model(x)
        repr_vec = bert_output.last_hidden_state #1*3*768
        return repr_vec

    def sample(self, q):
        rand = random.random()
        q = q.cpu().detach().numpy()
        for i in range(self.config.vocab_size):
            rand -= q[i]
            if (rand <= 0):
                return i

    def discriminate(self, seq):
        repr_vec = self.get_repr_vec(self.discriminator_model, seq)
        E = self.discriminator_linear(repr_vec[0,:])
        p_hat = torch.exp(-E)
        return p_hat

    def forward(self, input_seq):
        n = self.config.n
        k = self.config.k
        Pc0 = n/(n+k)
        rand = random.random()
        if (rand < Pc0):
            p_hat = self.discriminate(input_seq)
            q = self.generate(input_seq)
            loss = -torch.log(n*p_hat/(n*p_hat+k*q)) 
        else:
            generate_seq, q = self.generate()
            p_hat = self.discriminate(generate_seq)
            loss = -torch.log(k*q/(n*p_hat+k*q))
        return loss

    def look_up(self, triple, q_h, q_r, q_t):
        q_h = q_h[triple[0][0]]
        q_r = q_r[triple[0][1]]
        q_t = q_t[triple[0][2]]
        return q_h, q_r, q_t

    def triple2seq(self, triple):
        text = '[CLS]' + triple[0] + '[SEP]' + triple[1] + '[SEP]' + triple[2] + '[SEP]'
        seq = torch.tensor(self.tokenizer.encode(text))
        return seq

def train_and_eval(model, trainset, validset):
    model.train_mode()
    optimizer = optim.Adam(model.parameters(), lr=model.config.lr, weight_decay=model.config.weight_decay)
    train_num = trainset.shape[0]
    
    for e in range(model.config.epoch):
        permutation = np.random.permutation(train_num)
        shuffled_trainset = trainset[permutation, :]
        for b in range(train_num // model.module.config.batchsize):
            optimizer.zero_grad()
            for i in range(model.config.batchsize):
                triple = trainset[b*model.config.batchsize+i,:]
                seq = model.triple2seq(triple)
                input_tensor = seq.cuda(device=model.config.device_ids[0])
                if (i == 0):
                    loss = model(input_tensor)
                else:
                    loss += model(input_tensor)

            loss.backward()
            optimizer.step()
            logging.info("epoch "+str(e)+" step "+str(b)+" loss = "+str(loss.cpu().detach().numpy()[0]))

            if (b % 50 == 0):
                valid(model, validset)

def valid(model, validset):
    #model.module.eval_mode()
    valid_num = validset.shape[0]
    permutation = np.random.permutation(valid_num)
    shuffled_validset = validset[permutation, :]
    pair = []
    for i in range(model.config.valid_num):
        triple = validset[i,:]
        position = random.randint(0,2)
        id_true = triple[position]
        #triple[position] = 0
        input_triple = torch.tensor(triple).view([1,3]).cuda(device=model.config.device_ids[0])
        q_h, q_r, q_t = model.generate(input_triple)
        q = [q_h, q_r, q_t]
        q_list = q[position].cpu().detach().numpy()
        id_pred = q_list.argsort()[::-1]
        pair.append([id_true, id_pred])
    result = evaluate(pair)
    logging.info(result)
    #model.module.train_mode()  

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    trainset = np.load(os.path.join(args.data_dir, 'trainset.npy'))
    validset = np.load(os.path.join(args.data_dir, 'validset.npy'))

    logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                        filename='EBKG.log',
                        filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        #a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        #日志格式
                        )

    device_ids = [0,1,2,3]
    config = Config(device_ids=device_ids)
    ebkgm = EnergyBasedKGModel(config=config)
    #ebkgm_parallel = torch.nn.DataParallel(ebkgm, device_ids=ebkgm.config.device_ids)
    #ebkgm_parallel.to(ebkgm_parallel.module.main_device)
    ebkgm = ebkgm.cuda(device=device_ids[0])

    train_and_eval(ebkgm, trainset, validset)
    torch.save(ebkgm, os.path.join(args.model_dir,'checkpoint.pt'))

    pass

if __name__ == "__main__":
    main()
