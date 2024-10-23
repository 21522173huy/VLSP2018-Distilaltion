from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import yaml
import sys
import argparse
import torch
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch

def evaluate_model(model, test_dataloader, average = 'macro'):
    model.eval(), model.to('cuda')

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].float().to('cuda')
            labels = batch['labels'].to('cuda')

            # Predictions from the teacher model
            output = model(input_ids, attention_mask=attention_mask)
            # Prediction
            preds = output.argmax(dim=-1)

            # Flatten predictions and labels for evaluation
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()

            y_pred.extend(preds_flat.tolist())
            y_true.extend(labels_flat.tolist())

    if np.any(aspect_true) and np.any(aspect_pred):
        aspect_accuracy = accuracy_score(aspect_true, aspect_pred)
        aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(aspect_true, aspect_pred, average=average, zero_division=0)
    else:
        aspect_accuracy = aspect_precision = aspect_recall = aspect_f1 = 0

    # Sentiment Classification Metrics (only for correctly identified aspects)
    correct_aspects = [true and pred for true, pred in zip(aspect_true, aspect_pred)]
    if np.any(correct_aspects):
        sentiment_accuracy = accuracy_score([true for true, correct in zip(true_labels, correct_aspects) if correct],
                                            [pred for pred, correct in zip(teacher_predictions, correct_aspects) if correct])
        sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(
            [true for true, correct in zip(true_labels, correct_aspects) if correct],
            [pred for pred, correct in zip(teacher_predictions, correct_aspects) if correct],
            average=average, zero_division=0)
    else:
        sentiment_accuracy = sentiment_precision = sentiment_recall = sentiment_f1 = 0
    
    # Save results to a JSON file
    results = {
        "Aspect": {
            "accuracy_score": aspect_accuracy,
            "precision_score": aspect_precision,
            "recall_score": aspect_recall,
            "f1_score": aspect_f1
        },
        "Polarity": {
            "accuracy_score": sentiment_accuracy,
            "precision_score": sentiment_precision,
            "recall_score": sentiment_recall,
            "f1_score": sentiment_f1
        }
    }

    return results, y_pred, y_true
