"""Pre-trains an Energy-Based Knowledge Graph."""

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import re
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
                train_method='conditional_nce',
                negative_sample_num = 3,
                valid_num = 10,
                lr = 1e-6,
                weight_decay = 0,
                batchsize = 10,
                epoch = 10):
        self.device_ids = device_ids
        self.vocab_size=vocab_size
        #self.bert_config = bert_config
        self.train_method = train_method
        self.negative_sample_num = negative_sample_num
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
        #generator
        self.generator_name = "bert-base-uncased"#"albert-base-v1"#
        self.generator_config = BertConfig.from_pretrained(self.generator_name)
        self.generator_model = BertModel.from_pretrained(self.generator_name)
        self.generator_linear_h = torch.nn.Linear(self.generator_config.hidden_size, self.config.vocab_size)
        self.generator_linear_r = torch.nn.Linear(self.generator_config.hidden_size, self.config.vocab_size)
        self.generator_linear_t = torch.nn.Linear(self.generator_config.hidden_size, self.config.vocab_size)
        #discriminator
        self.discriminator_name = "bert-base-uncased"#"albert-base-v1"#
        self.discriminator_config = BertConfig.from_pretrained(self.discriminator_name)
        self.discriminator_model = BertModel.from_pretrained(self.discriminator_name)
        self.discriminator_linear_h = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
        self.discriminator_linear_r = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
        self.discriminator_linear_t = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
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

    def generate(self, input_triple):
        input_triple_mask_h = input_triple.clone()
        input_triple_mask_r = input_triple.clone()
        input_triple_mask_t = input_triple.clone()
        input_triple_mask_h[0][0] = 0
        input_triple_mask_r[0][1] = 0
        input_triple_mask_t[0][2] = 0
        repr_vec_h = self.get_repr_vec(self.generator_model, input_triple_mask_h)
        repr_vec_r = self.get_repr_vec(self.generator_model, input_triple_mask_r)
        repr_vec_t = self.get_repr_vec(self.generator_model, input_triple_mask_t)
        q_h = F.softmax(self.generator_linear_h(repr_vec_h[0,0,:]))
        q_r = F.softmax(self.generator_linear_r(repr_vec_r[0,1,:]))
        q_t = F.softmax(self.generator_linear_t(repr_vec_t[0,2,:]))
        return q_h, q_r, q_t

    def get_repr_vec(self, model, x):
        bert_output = model(x)
        repr_vec = bert_output.last_hidden_state #1*3*768
        return repr_vec

    def negative_sampling(self, input_triple, q_h, q_r, q_t):
        negative_samples = []
        for i in range(self.config.negative_sample_num):
            position = random.randint(0,2)
            resampled_triple = input_triple.clone()
            replaced = True
            if (position == 0):
                resampled_triple[0][0] = self.sample(q_h)#torch.argmax(q_h)
            if (position == 1):
                resampled_triple[0][1] = self.sample(q_r)#torch.argmax(q_r)
            if (position == 2):
                resampled_triple[0][2] = self.sample(q_t)#torch.argmax(q_t)
            if (resampled_triple.equal(input_triple)):
                replaced = False
            negative_samples.append({"triple":resampled_triple, "position":position, "replaced":replaced})
        return negative_samples
    
    def sample(self, q):
        rand = random.random()
        q = q.cpu().detach().numpy()
        for i in range(self.config.vocab_size):
            rand -= q[i]
            if (rand <= 0):
                return i

    def discriminate(self, triple):
        repr_vec = self.get_repr_vec(self.discriminator_model, triple)
        E_h = self.discriminator_linear_h(repr_vec[0,0,:])
        E_r = self.discriminator_linear_r(repr_vec[0,1,:])
        E_t = self.discriminator_linear_t(repr_vec[0,2,:])
        p_hat_h = torch.exp(-E_h)
        p_hat_r = torch.exp(-E_r)
        p_hat_t = torch.exp(-E_t)
        return p_hat_h, p_hat_r, p_hat_t

    def forward(self, input_triple):
        q_h, q_r, q_t = self.generate(input_triple)
        negative_samples = self.negative_sampling(input_triple, q_h, q_r, q_t)
        p_hat_h, p_hat_r, p_hat_t = self.discriminate(input_triple)
        q_h, q_r, q_t = self.look_up(input_triple, q_h, q_r, q_t)
        k = self.config.negative_sample_num
        loss = -torch.log(3*p_hat_h/(3*p_hat_h+k*q_h)) - torch.log(3*p_hat_r/(3*p_hat_r+k*q_r)) - torch.log(3*p_hat_t/(3*p_hat_t+k*q_t))
        for i in range(k):
            triple = negative_samples[i]["triple"]
            position = negative_samples[i]["position"]
            q_h, q_r, q_t = self.generate(triple)
            p_hat_h, p_hat_r, p_hat_t = self.discriminate(triple)
            q_h, q_r, q_t = self.look_up(triple, q_h, q_r, q_t)
            if (position == 0):
                loss -= torch.log(k*q_h/(3*p_hat_h+k*q_h))
            if (position == 1):
                loss -= torch.log(k*q_r/(3*p_hat_r+k*q_r))
            if (position == 2):
                loss -= torch.log(k*q_t/(3*p_hat_t+k*q_t))
        return loss

    def look_up(self, triple, q_h, q_r, q_t):
        q_h = q_h[triple[0][0]]
        q_r = q_r[triple[0][1]]
        q_t = q_t[triple[0][2]]
        return q_h, q_r, q_t
        
def train_and_eval(model, trainset, validset):
    model.module.train_mode()
    optimizer = optim.Adam(model.module.parameters(), lr=model.module.config.lr, weight_decay=model.module.config.weight_decay)
    train_num = trainset.shape[0]
    
    for e in range(model.module.config.epoch):
        permutation = np.random.permutation(train_num)
        shuffled_trainset = trainset[permutation, :]
        for b in range(train_num // model.module.config.batchsize):
            optimizer.zero_grad()
            first = True
            for i in range(model.module.config.batchsize):
                triple = torch.tensor(trainset[b*model.module.config.batchsize+i,:]).view([1,3])
                input_triple = triple.cuda(device=model.module.config.device_ids[0])
                if first:
                    loss = model(input_triple)
                    first = False
                else:
                    loss += model(input_triple)

            loss.backward()
            optimizer.step()
            print("epoch "+str(e)+" step "+str(b)+" loss = "+str(loss.cpu().detach().numpy()[0]))

            if (b % 50 == 0):
                valid(model, validset)

def valid(model, validset):
    #model.module.eval_mode()
    valid_num = validset.shape[0]
    permutation = np.random.permutation(valid_num)
    shuffled_validset = validset[permutation, :]
    pair = []
    for i in range(model.module.config.valid_num):
        triple = validset[i,:]
        position = random.randint(0,2)
        id_true = triple[position]
        #triple[position] = 0
        input_triple = torch.tensor(triple).view([1,3]).cuda(device=model.module.config.device_ids[0])
        q_h, q_r, q_t = model.module.generate(input_triple)
        q = [q_h, q_r, q_t]
        q_list = q[position].cpu().detach().numpy()
        id_pred = q_list.argsort()
        pair.append([id_true, id_pred])
    result = evaluate(pair)
    print(result)
    #model.module.train_mode()  

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    trainset = np.load(os.path.join(args.data_dir, 'trainset.npy'))
    validset = np.load(os.path.join(args.data_dir, 'validset.npy'))

    device_ids = [0,1,2,3]
    config = Config(device_ids=device_ids)
    ebkgm = EnergyBasedKGModel(config=config)
    ebkgm_parallel = torch.nn.DataParallel(ebkgm, device_ids=ebkgm.config.device_ids)
    #ebkgm_parallel.to(ebkgm_parallel.module.main_device)
    ebkgm_parallel = ebkgm_parallel.cuda(device=device_ids[0])

    train_and_eval(ebkgm_parallel, trainset, validset)
    torch.save(ebkgm, os.path.join(args.model_dir,'checkpoint.pt'))

    pass

if __name__ == "__main__":
    main()
