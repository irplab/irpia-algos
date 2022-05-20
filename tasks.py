import json

from celery import Celery
from celery.signals import worker_process_init

import numpy as np
import torch
from transformers import CamembertTokenizerFast, CamembertForSequenceClassification, Trainer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


LABEL_NAMES = ['scolomfr-voc-015-num-1179',
               'scolomfr-voc-015-num-1548',
               'scolomfr-voc-015-num-1032',
               'scolomfr-voc-015-num-1333',
               'scolomfr-voc-015-num-6360',
               'scolomfr-voc-015-num-980',
               'scolomfr-voc-015-num-1430',
               'scolomfr-voc-015-num-919',
               'scolomfr-voc-015-num-7755',
               'scolomfr-voc-015-num-1831',
               'scolomfr-voc-015-num-6364',
               'scolomfr-voc-015-num-1832',
               'scolomfr-voc-015-num-6365',
               'scolomfr-voc-015-num-1834',
               'scolomfr-voc-015-num-6369',
               'scolomfr-voc-015-num-7816']

MODEL_PATH = './models/checkpoint-10000'

app = Celery('tasks',
             broker='redis://localhost:6379/0',
             backend='rpc://')


def initialization():
    initialization.tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


    # Load trained model
    model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABEL_NAMES))
    # Define test trainer
    initialization.trainer = Trainer(model)


@worker_process_init.connect()
def setup(**kwargs):
    print('initializing domain-predict module')
    initialization()
    print('done initializing domain-predict module')


@app.task
def predict(title, description):
    tokenized_title = initialization.tokenizer([title],
                                 padding=True, truncation=True)
    dataset = Dataset(tokenized_title)
    # Make prediction
    raw_pred, a, b = initialization.trainer.predict(dataset)

    # Translate raw predictions
    pred = LABEL_NAMES[np.argmax(raw_pred, axis=1)[0]]

    return json.dumps({'domain': [pred]})
