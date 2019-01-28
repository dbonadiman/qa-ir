
import argparse
import json
import logging

import torch

from qair import models
from qair.data import utils
from qair.data.dataset import QAdataset
from qair.data.embeddings import Embeddings
from qair.evaluate import reranking
from qair.utils import train as train_utils


def trainer(name, config, dataset):
    if name is None:
        experiment_path = train_utils.timestamp_dir("results")
    else:    
        experiment_path = f'{name}'

    with open(utils.create_path(f'{experiment_path}/config.json'), 'w') as conff:
        json.dump(config, conff)
    logging.info(f'saving experiment in: {experiment_path}')
    train_utils.set_seed(config['seed'])
    logging.info('Loading embeddings..')

    vocab = Embeddings(f"data/embs/{dataset}/{config['embeddings']}.txt")
    logging.info('Initializing Net..')
    device = 'cuda'
    model = models.Model.by_name(config['model']['name'])(config['model']['params'], vocab, device).to(device)
    
    text_parser = models.Parser.by_name(config['parser'])(vocab)

    train_model = models.Trainer.by_name(config['train_as'])(text_parser, model)

    train_data = QAdataset(f'data/parsed/{dataset}/train.json')
    valid_data = QAdataset(f'data/parsed/{dataset}/dev.json')
    test_data = QAdataset(f'data/parsed/{dataset}/test.json')
    
    optimizer  = getattr(torch.optim, config['optimizer']['name'])(model.trainable_parameters(), **config['optimizer']['params'])
    train_model.fit(train_data, optimizer, validation=valid_data, save_point=f"{experiment_path}/{config['model']['name']}_test.pt", patience=config['patience'], batch_size=config['batch_size'], intervals=100)

    valid_pred = train_model.predict(valid_data)
    valid_pred.to_file(f"{experiment_path}/dev.json")
    test_pred = train_model.predict(test_data)
    test_pred.to_file(f"{experiment_path}/test.json")
    
    valid_metrics = reranking.evaluate(valid_pred, 0.5)
    test_metrics = reranking.evaluate(test_pred, 0.5)

    with open(f"{experiment_path}/valid_metrics_0.5.json", 'w') as f:
        json.dump(valid_metrics, f)
    with open(f"{experiment_path}/test_metrics_0.5.json", 'w') as f:
        json.dump(test_metrics, f)

    logging.info(f'Results on the valid set at treshold 0.5:\n{train_utils.print_metrics(valid_metrics)}')
    logging.info(f'Results on the test set at treshold 0.5: \n{train_utils.print_metrics(test_metrics)}')
  

    max_f1 = 0
    best_th = 0
    for i in range(1, 100):
        th = 1/100*i
        f1 = reranking.f1(valid_pred, th)
        if f1 > max_f1:
            max_f1 = f1
            best_th = th
    
    valid_metrics = reranking.evaluate(valid_pred, best_th)
    test_metrics = reranking.evaluate(test_pred, best_th)

    with open(f"{experiment_path}/valid_metrics_best.json", 'w') as f:
        json.dump(valid_metrics, f)
    with open(f"{experiment_path}/test_metrics_best.json", 'w') as f:
        json.dump(test_metrics, f)

    logging.info(f'Results on the validation set at treshold {best_th}:\n{train_utils.print_metrics(valid_metrics)}')
    logging.info(f'Results on the test set at treshold {best_th}: \n{train_utils.print_metrics(test_metrics)}')



if __name__=='__main__':
    train_utils.config_logger()
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='the config file for the model (in json format)')
    parser.add_argument("dataset", help="the dataset name", default='wikiqa')
    parser.add_argument("--override", help="json string to override some parameters")
    parser.add_argument("--name", help="the name of the experiment")
    args = parser.parse_args()
    cfg = train_utils.load_config(file_name=args.config, override=args.override) 
    logging.info(f'Current Model Config: {json.dumps(cfg, indent=4, sort_keys=True)}')
    trainer(args.name, cfg, args.dataset)
