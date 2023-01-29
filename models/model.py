import torch.nn as nn
import numpy as np
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.structure_model.structure_encoder import StructureEncoder
from models.text_feature_propagation import HiBBNTP

from transformers import BertModel


class HiBBN(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HiBBN, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device
        self.dataset = config.data.dataset

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.model_type = config.model.type

        if "bert" in self.model_type:
            self.bert = BertModel.from_pretrained("bert-base-cased")
            self.bert_dropout = nn.Dropout(0.1)
        else:
            self.token_embedding = EmbeddingLayer(
                vocab_map=self.token_map,
                embedding_dim=config.embedding.token.dimension,
                vocab_name='token',
                config=config,
                padding_index=vocab.padding_index,
                pretrained_dir=config.embedding.token.pretrained_file,
                model_mode=model_mode,
                initial_type=config.embedding.token.init_type
            )
            self.text_encoder = TextEncoder(config)

        self.structure_encoder = StructureEncoder(config=config,
                                                  label_map=vocab.v2i['label'],
                                                  device=self.device,
                                                  graph_model_type=config.structure_encoder.type)

        self.himatch = HiBBNTP(config=config,
                                   device=self.device,
                                   graph_model=self.structure_encoder,
                                   label_map=self.label_map,
                                   model_mode=model_mode,
                                   graph_model_label=None)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        if "bert" not in self.model_type:
            params.append({'params': self.text_encoder.parameters()})
            params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.himatch.parameters()})
        return params

    def forward(self, inputs, alpha):
        batch, mode = inputs[0], inputs[1]
        if mode == 'TRAIN':
            if "bert" in self.model_type:
                outputs1 = self.bert(batch['input_ids1'].to(self.config.train.device_setting.device),
                                    batch['segment_ids1'].to(self.config.train.device_setting.device),
                                    batch['input_mask1'].to(self.config.train.device_setting.device))
                pooled_output1 = outputs1[1]
                token_output1 = self.bert_dropout(pooled_output1)

                outputs2 = self.bert(batch['input_ids2'].to(self.config.train.device_setting.device),
                                    batch['segment_ids2'].to(self.config.train.device_setting.device),
                                    batch['input_mask2'].to(self.config.train.device_setting.device))
                pooled_output2 = outputs2[1]
                token_output2 = self.bert_dropout(pooled_output2)
            else:
                embedding1 = self.token_embedding(batch['token1'].to(self.config.train.device_setting.device))
                seq_len1 = batch['token_len1']
                token_output1 = self.text_encoder(embedding1, seq_len1)

                embedding2 = self.token_embedding(batch['token2'].to(self.config.train.device_setting.device))
                seq_len2 = batch['token_len2']
                token_output2 = self.text_encoder(embedding2, seq_len2)

            logits = self.himatch(token_output1, token_output2, alpha)
            return logits
        else:
            if "bert" in self.model_type:
                outputs = self.bert(batch['input_ids'].to(self.config.train.device_setting.device),
                                    batch['segment_ids'].to(self.config.train.device_setting.device),
                                    batch['input_mask'].to(self.config.train.device_setting.device))
                pooled_output = outputs[1]
                token_output = self.bert_dropout(pooled_output)
            else:
                embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
                seq_len = batch['token_len']
                token_output = self.text_encoder(embedding, seq_len)

            tmp = token_output.view(-1, 300)
            np.savetxt("wos_hibbn.txt", tmp.cpu().detach().numpy())
            logits = self.himatch(text_feature1=token_output, text_feature2=token_output, alpha=0.5)
            return logits
