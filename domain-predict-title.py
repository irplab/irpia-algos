import argparse
import json

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Irpia module for domain prediction from title.')
    parser.add_argument('--title', dest='title', default='',
                        help='Metadata title')
    parser.add_argument('--text', dest='text', default='',
                        help='Metadata text')
    return parser.parse_args()


def is_not_blank(s):
    return bool(s and not s.isspace())

if __name__ == '__main__':
    args = parse_arguments()

    if is_not_blank(args.title):
        tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

        X_test_tokenized = tokenizer([args.title],
                                     padding=True, truncation=True)

        dataset = Dataset(X_test_tokenized)
        # Load trained model
        model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABEL_NAMES))
        # Define test trainer
        trainer = Trainer(model)
        # Make prediction
        raw_pred, a, b = trainer.predict(dataset)

        # Preprocess raw predictions
        pred = LABEL_NAMES[np.argmax(raw_pred, axis=1)[0]]

        print(json.dumps({'domain': [pred]}))
    else:
        print(json.dumps({'domain': []}))
