
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class PolarityMapping:
    INDEX_TO_POLARITY = { 0: None, 1: 'positive', 2: 'negative', 3: 'neutral' }
    POLARITY_TO_INDEX = { None: 0, 'positive': 1, 'negative': 2, 'neutral': 3 }

class VLSPDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.mapping = PolarityMapping
        self.aspect_names = self.data.column_names[:-3]

    def __len__(self):
        return len(self.data)

    def num_labels(self):
        return len(self.aspect_names)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Labels
        labels = []
        for i, each in enumerate(self.aspect_names):
            labels.append(item[each])

        inputs = {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(labels)
        }

        return inputs

    def idx2polarity(self, label):
        res = []
        for i, each in enumerate(label):
            item = {
                'Aspect': self.aspect_names[i],
                'Polarity': self.mapping.INDEX_TO_POLARITY[each.item()]
            }
            res.append(item)
        return res
    
def create_dataset(data_path, batch_size=16):
    # Load the dataset
    dataset = load_from_disk(data_path)
    
    # Dataset
    train_dataset = VLSPDataset(dataset['train'])
    val_dataset = VLSPDataset(dataset['val'])
    test_dataset = VLSPDataset(dataset['test'])

    print('Train Dataset:', len(train_dataset))
    print('Val Dataset:', len(val_dataset))
    print('Test Dataset:', len(test_dataset))

    return train_dataset, val_dataset, test_dataset
