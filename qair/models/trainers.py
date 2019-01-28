import torch
import torch.nn as nn
import random
import logging


from qair.models import Trainer
from qair.evaluate import reranking


@Trainer.register('ranker')
class Ranker(Trainer):


    def process(self, dataset, mode='list', get_labels=False):
        if mode == 'list':
            dts = dataset.list
            objects = []
            labels = []
            for ex in dts:
                objects.append(self.parser.parse(ex))
                labels.append(ex.label)
        elif mode == 'rank':
            dts = dataset.rank()
            objects = []
            labels = []
            for pos, neg in dts:
                p_pos = self.parser.parse(pos)
                p_neg = self.parser.parse(neg)
                objects.append((p_pos, p_neg))
                objects.append((p_neg, p_pos))
                labels.append(1)
                labels.append(-1)
        if get_labels:
            return objects, labels
        else:
            return objects
    
    def predict(self, dataset, batch_size=200):
        with torch.no_grad():
            self.model.eval()
            p_dataset = self.process(dataset, 'list', get_labels=False)
            preds = []
            for batch in self.model.batched_iter(p_dataset, batch_size=batch_size):
                batch = self.parser.make_batch(batch, self.model.device)
                out = self.model.forward(batch)
                preds += [x for x in out]
            self.model.train()
            for i, pred in enumerate(preds):
                dataset.list[i].feat('prediction', pred.item())
        return dataset
    
    def fit(self,
            dataset,
            optimizer,
            batch_size=32,
            epochs=1000,
            validation=None,
            save_point='.models/bm.pt',
            patience=20,
            intervals=100):

        p_dataset, labels = self.process(dataset, 'rank', get_labels=True)
        data = list(zip(p_dataset, labels))
        best_map = 0.
        wait = 0.
        critereon = nn.MarginRankingLoss(0.5)

        for epoch in range(epochs):
            seen_batches = 0
            random.shuffle(data)
            running_loss = 0.
            for i, batch in enumerate(self.model.batched_iter(data, batch_size)):
                optimizer.zero_grad()
                data_batch, lbls = zip(*batch)
                docs1, docs2 = zip(*data_batch)
                docs1 = self.parser.make_batch(docs1, self.model.device)
                docs2 = self.parser.make_batch(docs2, self.model.device)
                lbls = torch.FloatTensor(lbls).unsqueeze(1).to(self.model.device)
                pred1 = self.model(docs1)
                pred2 = self.model(docs2)
                loss = critereon(pred1, pred2, lbls)
                running_loss += loss.item()
                if i%intervals == intervals-1:
                    valid_map = reranking.MAP(self.predict(validation))
                    if valid_map > best_map:
                        best_map = valid_map
                        wait = 0
                        self.model.checkpoint()
                    logging.info(f'Epoch {epoch} batch {i}: loss {running_loss/(i+1):.4f}, valid_acc {valid_map:.4f} best_acc {best_map:.4f}')
                loss.backward()
                optimizer.step()
                seen_batches = i
            wait += 1
            if wait >= patience:
                self.model.load_checkpoint()
                self.model.save(save_point)
                break
            valid_map = reranking.MAP(self.predict(validation))
            if valid_map > best_map:
                best_map = valid_map
                wait = 0
                self.model.checkpoint()
            logging.info(f'Epoch {epoch} loss {running_loss/(seen_batches+1):.4f}, valid_acc {valid_map:.4f} best_acc {best_map:.4f}')

@Trainer.register('classifier')
class Classifier(Trainer):



    def process(self, dataset, mode='list', get_labels=False):
        if mode == 'list':
            dts = dataset.list
            objects = []
            labels = []
            for ex in dts:
                objects.append(self.parser.parse(ex))
                labels.append(ex.label)
        if get_labels:
            return objects, labels
        else:
            return objects

    
    def predict(self, dataset, batch_size=200):
        with torch.no_grad():
            self.model.eval()
            p_dataset = self.process(dataset, 'list')
            preds = []
            for batch in self.model.batched_iter(p_dataset, batch_size=batch_size):
                batch    = self.parser.make_batch(batch, self.model.device)
                out      = torch.sigmoid(self.model(batch))
                preds   += [x for x in out]
            self.model.train()
            for i, pred in enumerate(preds):
                dataset.list[i].feat('prediction', pred.item())
        return dataset


    def fit(self, dataset, optimizer ,batch_size=32, epochs=1000, validation=None, save_point='.models/bm.pt', patience=20, intervals=100):
        processed_dataset, labels = self.process(dataset, 'list' ,True)
        data = list(zip(processed_dataset, labels))
        best_map = 0.
        wait = 0.
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            seen_batches = 0
            #random.shuffle(data)
            running_loss = 0.
            for i, batch in enumerate(self.model.batched_iter(data, batch_size)):
                optimizer.zero_grad()
                docs, lbls = zip(*batch)
                docs = self.parser.make_batch(docs, self.model.device)
                
                lbls = torch.FloatTensor(lbls).unsqueeze(1).to(self.model.device)
                pred = self.model(docs)
                loss = criterion(torch.sigmoid(pred), lbls)
                running_loss += loss.item()
                if i%intervals == intervals-1:
                    valid_map = reranking.MAP(self.predict(validation))
                    if valid_map > best_map:
                        best_map = valid_map
                        wait = 0
                        self.model.checkpoint()
                    logging.info(f'Epoch {epoch} batch {i}: loss {running_loss/(i+1):.4f}, valid_acc {valid_map:.4f}, best_acc {best_map:.4f}')
                loss.backward()
                optimizer.step()
                seen_batches = i
            
            valid_map = reranking.MAP(self.predict(validation))
            if valid_map > best_map:
                best_map = valid_map
                wait = 0
                self.model.checkpoint()
            wait += 1
            if wait >= patience:
                self.model.load_checkpoint()
                self.model.save(save_point)
                break

            logging.info(f'Epoch {epoch}: loss {running_loss/(seen_batches+1):.4f}, valid_acc {valid_map:.4f} best_acc {best_map:.4f}')
