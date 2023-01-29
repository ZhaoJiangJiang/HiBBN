import torch
import random
import numpy as np

class Collator(object):
    def __init__(self, config, vocab, mode="TRAIN"):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = config.train.device_setting.device
        self.label_size = len(vocab.v2i['label'].keys())
        self.mode = mode
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num
        self.negative_ratio = config.data.negative_ratio
        self.vocab = vocab

        self.version_higher_11 = True
        version = torch.__version__
        version_num = version.split('.')
        if (int(version_num[0]) == 1 and int(version_num[1]) <= 1) or int(version_num[0]) == 0:
            self.version_higher_11 = False

    def _multi_hot(self, batch_labels):
        """
        :param batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
        :return: multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
        """
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
        batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
        return batch_labels_multi_hot

    def __call__(self, batch):
        if self.mode == 'TRAIN':
            batch_token1 = []
            batch_label1 = []
            batch_doc_len1 = []
            batch_input_ids1 = []
            batch_input_mask1 = []
            batch_segment_ids1 = []
            batch_input_len1 = []

            batch_token2 = []
            batch_label2 = []
            batch_doc_len2 = []
            batch_input_ids2 = []
            batch_input_mask2 = []
            batch_segment_ids2 = []
            batch_input_len2 = []

            for sample_i, sample in enumerate(batch):
                batch_token1.append(sample['token1'])
                batch_label1.append(sample['label1'])
                batch_doc_len1.append(sample['token_len1'])
                batch_input_ids1.append(sample['input_ids1'])
                batch_input_mask1.append(sample['input_mask1'])
                batch_segment_ids1.append(sample['segment_ids1'])
                batch_input_len1.append(sample['input_len1'])

                batch_token2.append(sample['token2'])
                batch_label2.append(sample['label2'])
                batch_doc_len2.append(sample['token_len2'])
                batch_input_ids2.append(sample['input_ids2'])
                batch_input_mask2.append(sample['input_mask2'])
                batch_segment_ids2.append(sample['segment_ids2'])
                batch_input_len2.append(sample['input_len2'])

            batch_token1 = torch.tensor(batch_token1)
            batch_multi_hot_label1 = self._multi_hot(batch_label1)
            batch_doc_len1 = torch.FloatTensor(batch_doc_len1)
            batch_input_ids1 = torch.LongTensor(batch_input_ids1)
            batch_input_mask1 = torch.LongTensor(batch_input_mask1)
            batch_segment_ids1 = torch.LongTensor(batch_segment_ids1)
            batch_input_len1 = torch.LongTensor(batch_input_len1)

            batch_token2 = torch.tensor(batch_token2)
            batch_multi_hot_label2 = self._multi_hot(batch_label2)
            batch_doc_len2 = torch.FloatTensor(batch_doc_len2)
            batch_input_ids2 = torch.LongTensor(batch_input_ids2)
            batch_input_mask2 = torch.LongTensor(batch_input_mask2)
            batch_segment_ids2 = torch.LongTensor(batch_segment_ids2)
            batch_input_len2 = torch.LongTensor(batch_input_len2)

            batch_res = {
                'token1': batch_token1,
                'label1': batch_multi_hot_label1,
                'token_len1': batch_doc_len1,
                'label_list1': batch_label1,
                'input_ids1': batch_input_ids1,
                'input_mask1': batch_input_mask1,
                'segment_ids1': batch_segment_ids1,
                'input_len1': batch_input_len1,

                'token2': batch_token2,
                'label2': batch_multi_hot_label2,
                'token_len2': batch_doc_len2,
                'label_list2': batch_label2,
                'input_ids2': batch_input_ids2,
                'input_mask2': batch_input_mask2,
                'segment_ids2': batch_segment_ids2,
                'input_len2': batch_input_len2,
            }
            return batch_res
        else:
            batch_text = []
            batch_token = []
            batch_label = []
            batch_doc_len = []

            batch_input_ids = []
            batch_input_mask = []
            batch_segment_ids = []
            batch_input_len = []

            for sample_i, sample in enumerate(batch):
                batch_text.append(sample['text'])
                batch_token.append(sample['token'])
                batch_label.append(sample['label'])
                batch_doc_len.append(sample['token_len'])
                batch_input_ids.append(sample['input_ids'])
                batch_input_mask.append(sample['input_mask'])
                batch_segment_ids.append(sample['segment_ids'])
                batch_input_len.append(sample['input_len'])

            batch_token = torch.tensor(batch_token)
            batch_multi_hot_label = self._multi_hot(batch_label)
            batch_doc_len = torch.FloatTensor(batch_doc_len)

            batch_input_ids = torch.LongTensor(batch_input_ids)
            batch_input_mask = torch.LongTensor(batch_input_mask)
            batch_segment_ids = torch.LongTensor(batch_segment_ids)
            batch_input_len = torch.LongTensor(batch_input_len)

            batch_res = {
                'text': batch_text,
                'token': batch_token,
                'label': batch_multi_hot_label,
                'token_len': batch_doc_len,
                'label_list': batch_label,
                'input_ids': batch_input_ids,
                'input_mask': batch_input_mask,
                'segment_ids': batch_segment_ids,
                'input_len': batch_input_len
            }
            return batch_res