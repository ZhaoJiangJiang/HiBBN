from torch.utils.data import DataLoader

from data_modules.dataset import ClassificationDataset
from data_modules.collator import Collator


def data_loaders(config, vocab, data={'train': None, 'val': None, 'test': None, 'desc': None}, bert_tokenizer=None, only_train=False):
    on_memory = data['train'] is not None
    collate_fn_train = Collator(config, vocab, mode='TRAIN')
    collate_fn = Collator(config, vocab, mode='EVA')
    train_dataset = ClassificationDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'],
                                          bert_tokenizer=bert_tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn_train,
                              pin_memory=True)

    if only_train:
        return train_loader

    val_dataset = ClassificationDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'],
                                        bert_tokenizer=bert_tokenizer)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.eval.batch_size,
                            shuffle=True,
                            num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

    test_dataset = ClassificationDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'],
                                         bert_tokenizer=bert_tokenizer)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.eval.batch_size,
                             shuffle=False,
                             num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

    # new
    label_desc_dataset = ClassificationDataset(config, vocab, stage='DESC', on_memory=on_memory, corpus_lines=data['desc'],
                                               bert_tokenizer=bert_tokenizer)
    label_desc_loader = DataLoader(label_desc_dataset,
                                   batch_size=config.train.batch_size,
                                   shuffle=False,
                                   num_workers=config.train.device_setting.num_workers,
                                   collate_fn=collate_fn,
                                   pin_memory=True)

    return train_loader, val_loader, test_loader, label_desc_loader