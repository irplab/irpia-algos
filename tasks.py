import json
import os

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


from level_french_labels import level_french_labels

DOMAIN_LABEL_NAMES = ['scolomfr-voc-015-num-1179',
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

LEVEL_LABEL_NAMES = ['scolomfr-voc-022-num-004', 'scolomfr-voc-022-num-005', 'scolomfr-voc-022-num-006',
                     'scolomfr-voc-022-num-007', 'scolomfr-voc-022-num-010', 'scolomfr-voc-022-num-011',
                     'scolomfr-voc-022-num-013', 'scolomfr-voc-022-num-014', 'scolomfr-voc-022-num-015',
                     'scolomfr-voc-022-num-018', 'scolomfr-voc-022-num-020', 'scolomfr-voc-022-num-021',
                     'scolomfr-voc-022-num-023', 'scolomfr-voc-022-num-608', 'scolomfr-voc-022-num-129',
                     'scolomfr-voc-022-num-131', 'scolomfr-voc-022-num-132', 'scolomfr-voc-022-num-133',
                     'scolomfr-voc-022-num-134', 'scolomfr-voc-022-num-135', 'scolomfr-voc-022-num-136',
                     'scolomfr-voc-022-num-138', 'scolomfr-voc-022-num-201', 'scolomfr-voc-022-num-043',
                     'scolomfr-voc-022-num-044', 'scolomfr-voc-022-num-047', 'scolomfr-voc-022-num-048',
                     'scolomfr-voc-022-num-049', 'scolomfr-voc-022-num-083', 'scolomfr-voc-022-num-212',
                     'scolomfr-voc-022-num-027', 'scolomfr-voc-022-num-089', 'scolomfr-voc-022-num-090',
                     'scolomfr-voc-022-num-213', 'scolomfr-voc-022-num-095', 'scolomfr-voc-022-num-096',
                     'scolomfr-voc-022-num-097', 'scolomfr-voc-022-num-098', 'scolomfr-voc-022-num-153',
                     'scolomfr-voc-022-num-154', 'scolomfr-voc-022-num-099', 'scolomfr-voc-022-num-100',
                     'scolomfr-voc-022-num-103', 'scolomfr-voc-022-num-104', 'scolomfr-voc-022-num-288',
                     'scolomfr-voc-022-num-298', 'scolomfr-voc-022-num-231', 'scolomfr-voc-022-num-125',
                     'scolomfr-voc-022-num-126', 'scolomfr-voc-022-num-238', 'scolomfr-voc-022-num-640',
                     'scolomfr-voc-022-num-641', 'scolomfr-voc-022-num-650', 'scolomfr-voc-022-num-139',
                     'scolomfr-voc-022-num-146', 'scolomfr-voc-022-num-150', 'scolomfr-voc-022-num-151',
                     'scolomfr-voc-022-num-185', 'scolomfr-voc-022-num-187', 'scolomfr-voc-022-num-101',
                     'scolomfr-voc-022-num-102', 'scolomfr-voc-022-num-111', 'scolomfr-voc-022-num-112',
                     'scolomfr-voc-022-num-109', 'scolomfr-voc-022-num-110', 'scolomfr-voc-022-num-163',
                     'scolomfr-voc-022-num-164', 'scolomfr-voc-022-num-063']

levelid2label = {idx: label for idx, label in enumerate(LEVEL_LABEL_NAMES)}
levellabel2id = {label: idx for idx, label in enumerate(LEVEL_LABEL_NAMES)}

DOMAINS_MODEL_PATH = './models/checkpoint-10000'
LEVELS_MODEL_PATH = './models/checkpoint-39376'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Celery('tasks',
             broker=os.getenv("CELERY_BROCKER", "redis://redis-broker:6379/0"),
             backend=os.getenv("CELERY_BACKEND", "redis://redis-broker:6379/1"))


def initialization():
    initialization.tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base",
                                                                      device=device)

    # Load trained model
    domains_model = CamembertForSequenceClassification.from_pretrained(DOMAINS_MODEL_PATH,
                                                                       num_labels=len(DOMAIN_LABEL_NAMES)).to(device)
    levels_model = CamembertForSequenceClassification.from_pretrained(LEVELS_MODEL_PATH,
                                                                      problem_type="multi_label_classification",
                                                                      num_labels=len(LEVEL_LABEL_NAMES),
                                                                      id2label=levelid2label,
                                                                      label2id=levellabel2id).to(device)
    # Define test trainer
    initialization.domain_trainer = Trainer(domains_model)
    initialization.level_trainer = Trainer(levels_model)


@worker_process_init.connect()
def setup(**kwargs):
    print('initializing domain-predict module & level-predict module')
    initialization()
    print('done initializing domain-predict module & level-predict module')


@app.task
def predict_domain(title, description):
    if is_not_blank(f"{title} {description}"):
        tokenized_title = initialization.tokenizer([title],
                                                   padding=True, truncation=True)
        dataset = Dataset(tokenized_title)
        # Make prediction
        raw_pred, a, b = initialization.domain_trainer.predict(dataset)

        # Translate raw predictions
        pred = [DOMAIN_LABEL_NAMES[np.argmax(raw_pred, axis=1)[0]]]
    else:
        pred = []
    return json.dumps({'domain': pred})


def is_not_blank(s):
    return bool(s and not s.isspace())


@app.task
def predict_level(title, description):
    if is_not_blank(f"{title} {description}"):
        tokenized_title = initialization.tokenizer([title],
                                                   padding=True, truncation=True, max_length=128)
        dataset = Dataset(tokenized_title)
        # Make prediction
        raw_pred, a, b = initialization.level_trainer.predict(dataset)

        # Translate raw predictions
        preds = [LEVEL_LABEL_NAMES[int(index)] for index in list(np.where(raw_pred[0] > 0)[0])]
    else:
        preds = []

    print([level_french_labels[f'http://data.education.fr/voc/scolomfr/concept/{pred}'] for pred in preds])

    return json.dumps({'level': preds})
