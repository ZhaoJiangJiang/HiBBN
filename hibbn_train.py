import os
import sys
import torch
import random

import helper.logger as logger
from helper.configure import Configure
from data_modules.vocab import Vocab
from data_modules.data_loader import data_loaders
from models.model import HiBBN
from helper.adamw import AdamW
from helper.lr_schedulers import get_linear_schedule_with_warmup
from train_modules.trainer import Trainer
from train_modules.criterions import ClassificationLoss
from train_modules.criterions import ContrastLoss
from helper.utils import load_checkpoint, save_checkpoint

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertAdam

def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate, params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")

def train(config):
    corpus_vocab = Vocab(config, min_freq=5, max_size=config.vocabulary.max_token_vocab)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    train_loader, dev_loader, test_loader, label_desc_loader = data_loaders(config, corpus_vocab, bert_tokenizer=tokenizer)

    hibbn = HiBBN(config, corpus_vocab)
    hibbn.to(config.train.device_setting.device)

    if "bert" in config.model.type:
        t_total = int(len(train_loader) * (config.train.end_epoch - config.train.start_epoch))

        param_optimizer = list(hibbn.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(t_total * 0.1)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=config.train.optimizer.learning_rate, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=config.train.optimizer.learning_rate, schedule='warmup_linear', warmup=0.1, t_total=t_total)
        scheduler = None
    else:
        optimizer = set_optimizer(config, hibbn)
        scheduler = None

    criterion_cls = ClassificationLoss(loss_type="bce")
    criterion_cmp = ContrastLoss(loss_type="bce")

    trainer = Trainer(model=hibbn,
                      criterion=[criterion_cls, criterion_cmp],
                      optimizer=optimizer,
                      vocab=corpus_vocab,
                      tokenizer=tokenizer,
                      scheduler=scheduler,
                      config=config,
                      label_desc_loader=label_desc_loader,
                      train_loader=train_loader,
                      dev_loader=dev_loader,
                      test_loader=test_loader)

    model_checkpoint = config.train.checkpoint.dir
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
        latest_model_file = 'best_micro_HiMatch'
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance_dev, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                           model=hibbn,
                                                           config=config,
                                                           optimizer=optimizer)

            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance_dev[0], best_performance_dev[1]))

    # trainer.case_study()

    # trainer.run_train()

    hibbn.load_state_dict(torch.load("glove-wos_checkpoint/HiMatch.pt"))
    trainer.run_test()

if __name__ == '__main__':
    configs = Configure(config_json_file=sys.argv[1])

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    logger.Logger(configs)

    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    train(configs)