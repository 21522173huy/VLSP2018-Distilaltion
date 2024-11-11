
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation import evaluate_model
import torch
from early_stopping import EarlyStopping
import json
import numpy as np

def one_hot_encode(tensor):
    # Get the shape of the input tensor
    shape = tensor.shape
    
    # Create a new tensor for the one-hot encoded output
    one_hot = torch.zeros(*shape, 4, dtype=torch.float32)
    
    # Scatter the one-hot encoding
    one_hot.scatter_(2, tensor.cpu().unsqueeze(-1), 1.0)
    
    return one_hot.view(-1, shape[1] * 4)

def step(model, ver, dataloader, optimizer, criterion, device, max_grad_norm=1.0, mode='Train', average='macro'):
    if mode == 'Train': model.train()
    else: model.eval()

    total_loss = 0
    metrics_epoch = {
        'Aspect': {
            'accuracy': 0,
            'precision': 0, 
            'recall': 0, 
            'f1': 0
        },
        'Polarity': {
            'accuracy': 0, 
            'precision': 0, 
            'recall': 0, 
            'f1': 0
        }
    }

    for batch in tqdm(dataloader):
        
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].float().to(device)
        labels = batch['labels'].to(device)

        with torch.set_grad_enabled(mode == 'Train'):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if ver == 1:
              loss = criterion(output.mT, labels)
            else :
              one_hot_labels = one_hot_encode(labels).to(device)
              loss = criterion(output.to(device), one_hot_labels)

            # Backward
            if mode == 'Train':
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()    

        # Loss and Metrics
        total_loss += loss.item()
        
        # Prediction
        if ver == 1:
          preds = output.argmax(dim=-1)
        else : 
          reshaped_output = output.view(-1, model.num_labels, 4)
          preds = reshaped_output.argmax(dim=-1)

        # Flatten predictions and labels for evaluation
        preds_flat = preds.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()

        # Aspect Identification Metrics
        aspect_true = labels_flat != 0  # True aspects are those that are not 'None'
        aspect_pred = preds_flat != 0  # Predicted aspects are those that are not 'None'

        if np.any(aspect_true) and np.any(aspect_pred):
            aspect_accuracy = accuracy_score(aspect_true, aspect_pred)
            aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(aspect_true, aspect_pred, average=average, zero_division=0)
        else:
            aspect_accuracy = aspect_precision = aspect_recall = aspect_f1 = 0

        # Sentiment Classification Metrics (only for correctly identified aspects)
        correct_aspects = aspect_true & aspect_pred
        if np.any(correct_aspects):
            sentiment_accuracy = accuracy_score(labels_flat[correct_aspects], preds_flat[correct_aspects])
            sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(labels_flat[correct_aspects], preds_flat[correct_aspects], average=average, zero_division=0)
        else:
            sentiment_accuracy = sentiment_precision = sentiment_recall = sentiment_f1 = 0

        metrics_epoch['Aspect']['accuracy'] += aspect_accuracy
        metrics_epoch['Aspect']['precision'] += aspect_precision
        metrics_epoch['Aspect']['recall'] += aspect_recall
        metrics_epoch['Aspect']['f1'] += aspect_f1
        metrics_epoch['Polarity']['accuracy'] += sentiment_accuracy
        metrics_epoch['Polarity']['precision'] += sentiment_precision
        metrics_epoch['Polarity']['recall'] += sentiment_recall
        metrics_epoch['Polarity']['f1'] += sentiment_f1

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: {metric: v / len(dataloader) for metric, v in metrics.items()} for k, metrics in metrics_epoch.items()}

    return avg_loss, avg_metrics

def finetune_teacher(model, ver,
                     train_dataloader, val_dataloader, test_dataloader, 
                     optimizer, criterion, scheduler, epochs, 
                     checkpoint_path, result_path,
                     max_grad_norm=1.0, patience=5,
                     gradual_unfreezing = True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses, val_losses = [], []
    train_metrics_list, val_metrics_list = [], []

    best_f1_score = float(-1)
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    for epoch in range(epochs):
        
        # Gradual unFreezing
        if gradual_unfreezing:
            model.count_epochs()
        
        # Loss and Metrics
        print("Training")
        train_loss, train_metrics = step(model, ver, train_dataloader, optimizer, criterion, device, max_grad_norm, mode='Train')
        print("Validating")
        val_loss, val_metrics = step(model, ver, val_dataloader, optimizer, criterion, device, max_grad_norm, mode='Val')
        print('Testing')
        results, y_true, y_pred = evaluate_model(model, ver, test_dataloader, average = 'macro')

        # Obtain Loss and Metrics
        train_losses.append(train_loss)
        train_metrics_list.append(train_metrics)
        val_losses.append(val_loss)
        val_metrics_list.append(val_metrics)

        # Step the scheduler
        # scheduler.step(val_loss)
        scheduler.step()

        # Early Stopping
        early_stopping(val_loss, model)

        # Print statements
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Train Aspect Metrics:")
        print(f"  Accuracy: {train_metrics['Aspect']['accuracy']:.4f} | Precision: {train_metrics['Aspect']['precision']:.4f} | Recall: {train_metrics['Aspect']['recall']:.4f} | F1: {train_metrics['Aspect']['f1']:.4f}")
        print("Train Polarity Metrics:")
        print(f"  Accuracy: {train_metrics['Polarity']['accuracy']:.4f} | Precision: {train_metrics['Polarity']['precision']:.4f} | Recall: {train_metrics['Polarity']['recall']:.4f} | F1: {train_metrics['Polarity']['f1']:.4f}")

        print(f"\nValidation Loss: {val_loss:.4f}")
        print("Validation Aspect Metrics:")
        print(f"  Accuracy: {val_metrics['Aspect']['accuracy']:.4f} | Precision: {val_metrics['Aspect']['precision']:.4f} | Recall: {val_metrics['Aspect']['recall']:.4f} | F1: {val_metrics['Aspect']['f1']:.4f}")
        print("Validation Polarity Metrics:")
        print(f"  Accuracy: {val_metrics['Polarity']['accuracy']:.4f} | Precision: {val_metrics['Polarity']['precision']:.4f} | Recall: {val_metrics['Polarity']['recall']:.4f} | F1: {val_metrics['Polarity']['f1']:.4f}")

        print("\nResults in Test:")
        print("Test Aspect Metrics:")
        print(f"  Accuracy: {results['Aspect']['accuracy_score']:.4f} | Precision: {results['Aspect']['precision_score']:.4f} | Recall: {results['Aspect']['recall_score']:.4f} | F1: {results['Aspect']['f1_score']:.4f}")
        print("Test Polarity Metrics:")
        print(f"  Accuracy: {results['Polarity']['accuracy_score']:.4f} | Precision: {results['Polarity']['precision_score']:.4f} | Recall: {results['Polarity']['recall_score']:.4f} | F1: {results['Polarity']['f1_score']:.4f}")
        print(' ')

        # Save Best F1-score checkpoint
        if results['Polarity']['f1_score'] > best_f1_score:
            best_f1_score = results['Polarity']['f1_score']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
            
            with open(result_path, 'w') as f:
                json.dump({
                    'results': results,
                    'y_true': y_true,
                    'y_pred': y_pred,
                }, f, indent=4)

        # Stop
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses, train_metrics_list, val_metrics_list
