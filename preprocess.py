import argparse
import os
import re
import numpy as np

def fetch_id(file, type):
    list_ = []
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = re.split('[\t \n]', line)
            list_.append(type+line[1])
    return list_

def build_dataset(entity_file, relation_file, triple_file):
    all_entities = fetch_id(entity_file, 'e')
    all_relations = fetch_id(relation_file, 'r')
    entity_num = len(all_entities)
    relation_num = len(all_relations)
    vocab_size = entity_num + relation_num
    with open(triple_file, 'r', encoding='UTF-8') as f:
        all_triples = f.readlines()
        triple_num = len(all_triples)
        dataset = np.zeros([triple_num, 3])
        index = 0
        for triple in all_triples:
            triple = re.split('[\t \n]', triple)
            try:
                h = all_entities.index('e'+triple[0])
                r = all_relations.index('r'+triple[1])
                t = all_entities.index('e'+triple[2])
            except:
                continue
            dataset[index, :] = [h, r+entity_num, t]
            index += 1
        dataset = dataset[0:index, :]
    return dataset        


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    entity_file = os.path.join(args.data_dir, 'all_entities.txt')
    relation_file = os.path.join(args.data_dir, 'all_relations.txt')
    train_triple_file = os.path.join(args.data_dir, 'train_triples.txt')
    valid_triple_file = os.path.join(args.data_dir, 'valid_triples.txt')
    test_triple_file = os.path.join(args.data_dir, 'test_triples.txt')
    trainset = build_dataset(entity_file=entity_file, relation_file=relation_file, triple_file=train_triple_file).astype(int)
    validset = build_dataset(entity_file=entity_file, relation_file=relation_file, triple_file=valid_triple_file).astype(int)
    testset = build_dataset(entity_file=entity_file, relation_file=relation_file, triple_file=test_triple_file).astype(int)
    np.save(os.path.join(args.data_dir, 'trainset.npy'), trainset)
    np.save(os.path.join(args.data_dir, 'validset.npy'), validset)
    np.save(os.path.join(args.data_dir, 'testset.npy'), testset)

if __name__ == "__main__":
    main()