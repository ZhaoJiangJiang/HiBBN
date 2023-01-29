import os
import json
import random
from collections import defaultdict
from torch.utils.data.dataset import Dataset

import helper.logger as logger

def get_sample_position(corpus_filename, on_memory, corpus_lines, stage):
    """
    position of each sample in the original corpus File or on-memory List
    :param corpus_filename: Str, directory of the corpus file
    :param on_memory: Boolean, True or False
    :param corpus_lines: List[Str] or None, on-memory Data
    :param mode: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
    :return: sample_position -> List[int]
    """
    sample_position = [0]
    if not on_memory:
        print('Loading files for ' + stage + ' Dataset...')
        with open(corpus_filename, 'r') as f_in:
            sample_str = f_in.readline()
            while sample_str:
                sample_position.append(f_in.tell())
                sample_str = f_in.readline()
            sample_position.pop()
    else:
        assert corpus_lines
        sample_position = range(len(corpus_lines))
    return sample_position

def get_sample(corpus_filename):
    data = []
    cate_have = defaultdict(list)
    cate_count = defaultdict(int)
    with open(corpus_filename, "r") as f_in:
        item = 0
        for line in f_in.readlines():
            line = json.loads(line.strip())
            for cate in line['label']:
                cate_have[cate].append(item)
                cate_count[cate] += 1
            data.append(line)
            item += 1
    return data, cate_have, cate_count

class ClassificationDataset(Dataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=True, corpus_lines=None, mode="TRAIN", bert_tokenizer=None):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        :param on_memory: Boolean, True or False
        :param corpus_lines: List[Str] or None, on-memory Data
        :param mode: TRAIN / PREDICT, for loading empty label
        """
        super(ClassificationDataset, self).__init__()
        # new
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file),
                             "DESC": os.path.join(config.data.data_dir, config.data.label_desc_file)}
        self.config = config
        self.vocab = vocab
        self.label_list = list(vocab.v2i['label'].keys())
        self.on_memory = on_memory
        self.max_input_length = self.config.text_encoder.max_length
        self.corpus_file = self.corpus_files[stage]
        self.data, self.cate_have, self.cate_count = get_sample(self.corpus_file)
        self.cate_count = dict(sorted(self.cate_count.items(), key=lambda x: x[1], reverse=True))
        self.sample_position = get_sample_position(self.corpus_file, self.on_memory, corpus_lines, stage)
        self.corpus_size = len(self.sample_position)
        self.mode = mode
        self.stage = stage
        self.sample_num = config.data.positive_num
        self.negative_ratio = config.data.negative_ratio
        self.dataset = config.data.dataset
        self.tokenizer = bert_tokenizer

        self.cate_keys = list(self.cate_count.keys())
        n_max = list(self.cate_count.values())[0]
        w = []
        for c in self.cate_keys:
            w.append(n_max / self.cate_count[c])
        self.reverse_p = []
        for i in range(len(w)):
            self.reverse_p.append(w[i] / sum(w))

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def reversed_sampling(self):
        sums = 0
        rand = random.random()
        for c, p in zip(self.cate_keys, self.reverse_p):
            sums += p
            if rand < sums:
                return c

    def __getitem__(self, item):
        """
        sample from the overall corpus
        :param item: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if item >= self.__len__():
            raise IndexError
        sample_str = self.data[item]

        contrast_cate = self.reversed_sampling()
        while len(self.cate_have[contrast_cate]) == 0:
            contrast_cate = self.reversed_sampling()
        contrast_item = random.choice(self.cate_have[contrast_cate])
        contrast_str = self.data[contrast_item]
        return self._preprocess_sample(sample_str, contrast_str)

    def create_features(self, sentences, max_seq_len=256, mode='1'):
        tokens_a = self.tokenizer.tokenize(sentences)
        tokens_b = None

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        if mode == '0':
            feature = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'input_len': input_len}
        elif mode == '1':
            feature = {'input_ids1': input_ids, 'input_mask1': input_mask, 'segment_ids1': segment_ids, 'input_len1': input_len}
        else:
            feature = {'input_ids2': input_ids, 'input_mask2': input_mask, 'segment_ids2': segment_ids, 'input_len2': input_len}
        return feature

    def _preprocess_sample(self, sample_str, contrast_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int, 'positive_sample': List[int], 'negative_sample': List[int], 'margin': List[int]}
        """
        if self.stage == 'TRAIN':
            raw_sample = sample_str
            sample = {'token1': [], 'label1': [], 'token2': [], 'label2': []}
            for k in raw_sample.keys():
                if k == 'token':
                    sample['token1'] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]
                    sentences = " ".join(raw_sample[k])
                    features = self.create_features(sentences, self.max_input_length, mode='1')
                    for (features_k, features_v) in features.items():
                        sample[features_k] = features_v
                elif k == 'label':
                    sample['label1'] = []
                    for v in raw_sample[k]:
                        if v not in self.vocab.v2i[k].keys():
                            logger.warning('Vocab not in ' + k + ' ' + v)
                        else:
                            sample['label1'].append(self.vocab.v2i[k][v])
            if not sample['token1']:
                sample['token1'].append(self.vocab.padding_index)
            sample['token_len1'] = min(len(sample['token1']), self.max_input_length)
            padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token1']))]
            sample['token1'] += padding
            sample['token1'] = sample['token1'][:self.max_input_length]

            raw_contrast = contrast_str
            for k in raw_contrast.keys():
                if k == 'token':
                    sample['token2'] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_contrast[k]]
                    sentences = " ".join(raw_contrast[k])
                    features = self.create_features(sentences, self.max_input_length, mode='2')
                    for (features_k, features_v) in features.items():
                        sample[features_k] = features_v
                elif k == 'label':
                    sample['label2'] = []
                    for v in raw_contrast[k]:
                        if v not in self.vocab.v2i[k].keys():
                            logger.warning('Vocab not in ' + k + ' ' + v)
                        else:
                            sample['label2'].append(self.vocab.v2i[k][v])
            if not sample['token2']:
                sample['token2'].append(self.vocab.padding_index)
            sample['token_len2'] = min(len(sample['token2']), self.max_input_length)
            padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token2']))]
            sample['token2'] += padding
            sample['token2'] = sample['token2'][:self.max_input_length]

            return sample
        else:
            raw_sample = sample_str
            sample = {'text': [], 'token': [], 'label': []}
            for k in raw_sample.keys():
                if k == 'token':

                    sample['text'] = raw_sample['token']
                    sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]

                    sentences = " ".join(raw_sample[k])
                    features = self.create_features(sentences, self.max_input_length, mode='0')
                    for (features_k, features_v) in features.items():
                        sample[features_k] = features_v

                elif k == 'label':
                    sample[k] = []
                    for v in raw_sample[k]:
                        if v not in self.vocab.v2i[k].keys():
                            logger.warning('Vocab not in ' + k + ' ' + v)
                        else:
                            sample[k].append(self.vocab.v2i[k][v])

            if not sample['token']:
                sample['token'].append(self.vocab.padding_index)
            sample['token_len'] = min(len(sample['token']), self.max_input_length)
            padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
            sample['token'] += padding
            sample['token'] = sample['token'][:self.max_input_length]
            return sample
