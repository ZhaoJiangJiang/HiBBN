import os
import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn


class HiBBNTP(nn.Module):
    def __init__(self, config, label_map, graph_model, device, model_mode, graph_model_label=None):
        """

        :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        :param model_mode:
        :param graph_model_label: computational graph for label graph
        """
        super(HiBBNTP, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map

        self.graph_model = graph_model
        self.graph_model_label = graph_model_label
        self.dataset = config.data.dataset

        self.linear1 = nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension,
                                len(self.label_map))

        self.linear2 = nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension,
                                len(self.label_map))

        # linear transform
        self.transformation = nn.Linear(config.model.linear_transformation.text_dimension,
                                        len(self.label_map) * config.model.linear_transformation.node_dimension)

        # dropout
        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)


    def forward(self, text_feature1, text_feature2, alpha):
        text_feature1 = text_feature1.view(text_feature1.shape[0], -1)
        text_feature2 = text_feature2.view(text_feature2.shape[0], -1)

        text_feature1 = alpha * text_feature1
        text_feature2 = (1-alpha) * text_feature2

        text_feature1 = self.transformation_dropout(self.transformation(text_feature1))
        text_feature1 = text_feature1.view(text_feature1.shape[0],
                                           len(self.label_map),
                                           self.config.model.linear_transformation.node_dimension)

        label_wise_text_feature1 = self.graph_model(text_feature1)

        logits1 = self.linear1(label_wise_text_feature1.view(label_wise_text_feature1.shape[0], -1))
        if self.config.model.classifier.output_drop:
            logits1 = self.dropout(logits1)

        text_feature2 = self.transformation_dropout(self.transformation(text_feature2))
        text_feature2 = text_feature2.view(text_feature2.shape[0],
                                           len(self.label_map),
                                           self.config.model.linear_transformation.node_dimension)

        label_wise_text_feature2 = self.graph_model(text_feature2)

        logits2 = self.linear2(label_wise_text_feature2.view(label_wise_text_feature2.shape[0], -1))
        if self.config.model.classifier.output_drop:
            logits2 = self.dropout(logits2)

        logits = logits1 + logits2

        return logits

        # if mode == 'TRAIN':
        #     text_feature1 = text_feature1.view(text_feature1.shape[0], -1)
        #     text_feature2 = text_feature2.view(text_feature2.shape[0], -1)
        #
        #     text_feature1 = self.transformation_dropout(self.transformation(text_feature1))
        #     text_feature1 = text_feature1.view(text_feature1.shape[0],
        #                                      len(self.label_map),
        #                                      self.config.model.linear_transformation.node_dimension)
        #
        #     label_wise_text_feature1 = self.graph_model(text_feature1)
        #
        #     logits1 = self.linear1(label_wise_text_feature1.view(label_wise_text_feature1.shape[0], -1))
        #     if self.config.model.classifier.output_drop:
        #         logits1 = self.dropout(logits1)
        #
        #     text_feature2 = self.transformation_dropout(self.transformation(text_feature2))
        #     text_feature2 = text_feature2.view(text_feature2.shape[0],
        #                                      len(self.label_map),
        #                                      self.config.model.linear_transformation.node_dimension)
        #
        #     label_wise_text_feature2 = self.graph_model(text_feature2)
        #
        #     logits2 = self.linear2(label_wise_text_feature2.view(label_wise_text_feature2.shape[0], -1))
        #     if self.config.model.classifier.output_drop:
        #         logits2 = self.dropout(logits2)
        #
        #     return logits1, logits2
        #
        # else:
        #     text_feature = text_feature1
        #     text_feature = text_feature.view(text_feature.shape[0], -1)
        #     text_feature = self.transformation_dropout(self.transformation(text_feature))
        #     text_feature = text_feature.view(text_feature.shape[0],
        #                                      len(self.label_map),
        #                                      self.config.model.linear_transformation.node_dimension)
        #
        #     label_wise_text_feature = self.graph_model(text_feature)
        #
        #     logits = self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1))
        #     if self.config.model.classifier.output_drop:
        #         logits = self.dropout(logits)
        #     return logits