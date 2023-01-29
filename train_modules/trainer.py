import os
import tqdm
import torch
import random
import numpy as np

import helper.logger as logger
from helper.utils import load_checkpoint, save_checkpoint
from train_modules.evaluation_metrics import evaluate
from train_modules.evaluation_metrics import evaluate_wos_layer
from train_modules.evaluation_metrics import evaluate_nyt_layer
from train_modules.evaluation_metrics import evaluate_rcv_layer
from train_modules.evaluation_metrics import case_study_nyt
from data_modules.data_loader import data_loaders


# for debug
torch.autograd.set_detect_anomaly(True)

class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, vocab, tokenizer, config, label_desc_loader=None,
                 train_loader=None, dev_loader=None, test_loader=None):
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        # self.criterion, self.criterion_ranking = criterion[0], criterion[1]
        self.criterion_cls, self.criterion_cmp = criterion[0], criterion[1]
        self.optimizer = optimizer
        self.dataset = config.data.dataset
        self.scheduler = scheduler
        self.eval_steps = config.eval.eval_steps

        self.ckpt_path = "{}/{}.pt".format(self.config.train.checkpoint.dir, self.config.model.type)

        self.train_end_epoch = config.train.end_epoch
        self.train_start_epoch = config.train.start_epoch
        self.begin_eval_epoch = config.eval.begin_eval_epoch
        self.refresh_data_loader = config.train.refresh_data_loader

        self.tokenizer = tokenizer
        self.model_type = config.model.type

        self.best_epoch_dev = [-1, -1]
        self.best_performance_dev = [-0.01, -0.01]
        self.best_performance_test = [-0.01, -0.01]
        self.global_step = 0

        self.label_desc_loader = label_desc_loader
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.loader_map = {"TRAIN": self.train_loader, "DEV": self.dev_loader, "TEST": self.test_loader}

        self.case_study_path = "{}/{}_case_study.csv".format(self.config.train.checkpoint.dir, self.config.model.type)

    def update_lr(self):
        logger.warning('Learning rate update {}--->{}'.format(
            self.optimizer.param_groups[0]['lr'],
            self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def case_study(self, mode='TEST'):
        input_tokens = []
        predict_probs = []
        target_labels = []
        self.model.eval()
        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            input_tokens.extend(batch['text'])
            # logits = self.model([batch, mode, label_repre])
            logits = self.model([batch, mode], alpha=0.5)

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        case_study_nyt(input_tokens, predict_probs, target_labels, self.vocab, self.case_study_path, self.config.eval.threshold)

    def run_test(self, mode='TEST'):
        predict_probs = []
        target_labels = []
        self.model.eval()
        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            # logits = self.model([batch, mode, label_repre])
            logits = self.model([batch, mode], alpha=0.5)

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        if self.config.data.dataset == 'wos':
            hier_p = evaluate_wos_layer(predict_probs, target_labels, self.vocab, self.config.eval.threshold)
            logger.info("precision = {}".format(hier_p['precision']))
            logger.info("recall = {}".format(hier_p['recall']))
            logger.info("micro_f1 = {}".format(hier_p['micro_f1']))
            logger.info("macro_f1 = {}".format(hier_p['macro_f1']))
            logger.info("l1_micro_f1 = {}".format(hier_p['l1_micro_f1']))
            logger.info("l2_micro_f1 = {}".format(hier_p['l2_micro_f1']))
            logger.info("l1_macro_f1 = {}".format(hier_p['l1_macro_f1']))
            logger.info("l2_macro_f1 = {}".format(hier_p['l2_macro_f1']))

        elif self.config.data.dataset == 'rcv1v2':
            hier_p = evaluate_rcv_layer(predict_probs, target_labels, self.vocab, self.config.eval.threshold)
            logger.info("precision = {}".format(hier_p['precision']))
            logger.info("recall = {}".format(hier_p['recall']))
            logger.info("micro_f1 = {}".format(hier_p['micro_f1']))
            logger.info("macro_f1 = {}".format(hier_p['macro_f1']))
            logger.info("l1_micro_f1 = {}".format(hier_p['l1_micro_f1']))
            logger.info("l2_micro_f1 = {}".format(hier_p['l2_micro_f1']))
            logger.info("l3_micro_f1 = {}".format(hier_p['l3_micro_f1']))
            logger.info("l1_macro_f1 = {}".format(hier_p['l1_macro_f1']))
            logger.info("l2_macro_f1 = {}".format(hier_p['l2_macro_f1']))
            logger.info("l3_macro_f1 = {}".format(hier_p['l3_macro_f1']))

        elif self.config.data.dataset == 'nyt':
            hier_p = evaluate_nyt_layer(predict_probs, target_labels, self.vocab, self.config.eval.threshold)
            logger.info("precision = {}".format(hier_p['precision']))
            logger.info("recall = {}".format(hier_p['recall']))
            logger.info("micro_f1 = {}".format(hier_p['micro_f1']))
            logger.info("macro_f1 = {}".format(hier_p['macro_f1']))
            logger.info("l1_micro_f1 = {}".format(hier_p['l1_micro_f1']))
            logger.info("l2_micro_f1 = {}".format(hier_p['l2_micro_f1']))
            logger.info("l3_micro_f1 = {}".format(hier_p['l3_micro_f1']))
            logger.info("l1_macro_f1 = {}".format(hier_p['l1_macro_f1']))
            logger.info("l2_macro_f1 = {}".format(hier_p['l2_macro_f1']))
            logger.info("l3_macro_f1 = {}".format(hier_p['l3_macro_f1']))

    def run_eval(self, epoch, mode="EVAL"):
        predict_probs = []
        target_labels = []
        label_repre = -1

        self.model.eval()
        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            # logits = self.model([batch, mode, label_repre])
            logits = self.model([batch, mode], alpha=0.5)

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        performance = evaluate(predict_probs, target_labels, self.vocab, self.config.eval.threshold)
        if performance['micro_f1'] > self.best_performance_dev[0]:
            logger.info('DEV Improve Micro-F1 {}% --> {}%'.format(self.best_performance_dev[0], performance['micro_f1']))
            self.best_performance_dev[0] = performance['micro_f1']
            self.best_epoch_dev[0] = epoch
            logger.info('Achieve best Micro-F1 on dev set, evaluate on test set')
            torch.save(self.model.state_dict(), self.ckpt_path)

            # test_performance = self.run_eval(epoch, "EVAL")
            # if test_performance['micro_f1'] > self.best_performance_test[0]:
            #     logger.info('TEST Improve Micro-F1 {}% --> {}%'.format(self.best_performance_test[0], test_performance['micro_f1']))
            #     self.best_performance_test[0] = test_performance['micro_f1']
            #     torch.save(self.model.state_dict(), self.ckpt_path)

        if performance['macro_f1'] > self.best_performance_dev[1]:
            logger.info('DEV Improve Macro-F1 {}% --> {}%'.format(self.best_performance_dev[1], performance['macro_f1']))
            self.best_performance_dev[1] = performance['macro_f1']
            self.best_epoch_dev[1] = epoch
            logger.info('Achieve best Macro-F1 on dev set, evaluate on test set')
            torch.save(self.model.state_dict(), self.ckpt_path)

            # test_performance = self.run_eval(epoch, "EVAL")
            # if test_performance['macro_f1'] > self.best_performance_test[1]:
            #     logger.info('TEST Improve Macro-F1 {}% --> {}%'.format(self.best_performance_test[1], test_performance['macro_f1']))
            #     self.best_performance_test[1] = test_performance['macro_f1']
            #     torch.save(self.model.state_dict(), self.ckpt_path)
        return performance

    def run_train(self, mode='TRAIN'):
        total_epoch = self.config.train.end_epoch
        for epoch in range(self.train_start_epoch, self.train_end_epoch):
            alpha = 1 - (epoch / total_epoch) ** 2
            # alpha = 1 - epoch / total_epoch
            if self.refresh_data_loader == 1:
                self.train_loader = data_loaders(self.config, self.vocab, bert_tokenizer=self.tokenizer, only_train=True)

            predict_probs = []
            target_labels = []
            total_loss = 0.0
            label_repre = -1
            if epoch % 10 == 0:
                logger.info("========== epoch: %d ==========" % epoch)
                if epoch == 0:
                    logger.info("begin evaluation after epoch: %d" % self.begin_eval_epoch)

            for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
                logits = self.model([batch, mode], alpha)
                loss1 = self.criterion_cls(logits, batch['label1'].to(self.config.train.device_setting.device))
                loss2 = self.criterion_cls(logits, batch['label2'].to(self.config.train.device_setting.device))
                total_loss += loss1.item() + loss2.item()

                loss = alpha*loss1 + (1-alpha)*loss2
                self.optimizer.zero_grad()

                loss.backward(retain_graph=True)

                """如果不是使用BertAdam的话 这里要打开"""
                # if "bert" in self.config.model.type:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                #     self.scheduler.step()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if (self.global_step + 1) % self.eval_steps == 0 and epoch >= self.begin_eval_epoch:
                    performance = self.run_eval(epoch, "DEV")
                    self.model.train()
                    # self.model.load_state_dict(torch.load(self.ckpt_path))

                # predict_results = torch.sigmoid(logits).cpu().tolist()
                # predict_probs.extend(predict_results)
                # target_labels.extend(batch['label_list'])
            logger.info("loss: %f alpha: %f" % (total_loss / len(self.loader_map[mode]), alpha))

        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.run_test()


