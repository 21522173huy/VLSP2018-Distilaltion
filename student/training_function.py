from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation import evaluate_model
import torch
from torch import nn
from early_stopping import EarlyStopping
import torch.nn.functional as F
import numpy as np  # Add missing import
import json

def check_and_set_requires_grad(tensor):
    if not tensor.requires_grad:
        tensor.requires_grad_(True)
    return tensor

def one_hot_encode(tensor):
    # Get the shape of the input tensor
    shape = tensor.shape
    
    # Create a new tensor for the one-hot encoded output
    one_hot = torch.zeros(*shape, 4, dtype=torch.float32)
    
    # Scatter the one-hot encoding
    one_hot.scatter_(2, tensor.cpu().unsqueeze(-1), 1.0)
    
    return one_hot.view(-1, shape[1] * 4)

def single_label_loss(student_logits, teacher_logits, labels, temperature, soft_weight, hard_weight):
    """
    Compute the distillation loss.
    
    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        true_labels (torch.Tensor): True labels.  # Fix docstring parameter name
        alpha (float): Weight for the soft loss.  # Fix docstring parameter name
    
    Returns:
        torch.Tensor: The computed distillation loss.
    """
    # Soft targets
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets
    hard_loss = F.cross_entropy(student_logits.mT, labels)
    
    # Combined loss
    loss = soft_weight * soft_loss + hard_weight * hard_loss
    return loss

def multi_label_loss(student_logits, teacher_logits, one_hot_labels, temperature, soft_weight, hard_weight):
    """
    Compute the distillation loss for multi-label classification.
    
    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        true_labels (torch.Tensor): True labels.  # Fix docstring parameter name
        alpha (float): Weight for the soft loss.  # Fix docstring parameter name
    
    Returns:
        torch.Tensor: The computed distillation loss.
    """
    # Soft targets
    soft_targets = F.sigmoid(teacher_logits / temperature)
    soft_loss = F.kl_div(F.logsigmoid(student_logits / temperature), soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, one_hot_labels)
    
    # Combined loss
    loss = soft_weight * soft_loss + hard_weight * hard_loss
    return loss

def step(student_model, teacher_model, ver, dataloader, optimizer, device, temperature=2.0, soft_weight=0.5, hard_weight=0.5, pred_distill=False, max_grad_norm=1.0, mode='Train', average='micro'):
    teacher_model.eval()
    if mode == 'Train': student_model.train()
    else: student_model.eval()

    criterion = nn.MSELoss()

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

        if mode == 'Train': optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'Train'):
            student_logits, student_attns, student_hidden_states = student_model(input_ids=input_ids, attention_mask=attention_mask)
            with torch.no_grad():
                teacher_logits, teacher_attns, teacher_hidden_states = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate losses
            if not pred_distill:
                teacher_layer_num = len(teacher_attns)
                student_layer_num = len(student_attns)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)

                new_teacher_atts = [teacher_attns[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]
                new_teacher_reps = [teacher_hidden_states[i * layers_per_block] for i in range(student_layer_num + 1)]

                # Replace values fewer than -1e2 by 0
                student_attns = [torch.where(att <= -1e2, torch.zeros_like(att).to(device), att) for att in student_attns]
                new_teacher_atts = [torch.where(att <= -1e2, torch.zeros_like(att).to(device), att) for att in new_teacher_atts]

                # Concatenate all attention and hidden states
                student_attns_concat = torch.cat([att.view(-1) for att in student_attns])
                new_teacher_atts_concat = torch.cat([att.view(-1) for att in new_teacher_atts])
                student_hidden_states_concat = torch.cat([rep.view(-1) for rep in student_hidden_states])
                new_teacher_reps_concat = torch.cat([rep.view(-1) for rep in new_teacher_reps])

                # Require grad for all tensors
                student_attns_concat = check_and_set_requires_grad(student_attns_concat)
                new_teacher_atts_concat = check_and_set_requires_grad(new_teacher_atts_concat)
                student_hidden_states_concat = check_and_set_requires_grad(student_hidden_states_concat)
                new_teacher_reps_concat = check_and_set_requires_grad(new_teacher_reps_concat)

                # Calculate MSE loss
                att_loss = criterion(student_attns_concat, new_teacher_atts_concat)
                rep_loss = criterion(student_hidden_states_concat, new_teacher_reps_concat)

                loss = rep_loss + att_loss

            else:
                if ver == 1: # Batch_size, Num_labels, 4
                    loss = single_label_loss(student_logits, teacher_logits, labels, temperature, soft_weight, hard_weight)
                else:
                    one_hot_labels = one_hot_encode(labels).to(device)
                    loss = multi_label_loss(student_logits, teacher_logits, one_hot_labels, temperature, soft_weight, hard_weight)

            # Backward
            if mode == 'Train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                optimizer.step()

        # Loss and Metrics
        total_loss += loss.item()

        # Prediction
        if ver == 1:
          preds = student_logits.argmax(dim=-1)
        else : 
          reshaped_output = student_logits.view(-1, student_model.num_labels, 4)
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

def training_student(pred_distill,
                     student_model, teacher_model, ver,
                     train_dataloader, val_dataloader, test_dataloader,
                     optimizer, scheduler, epochs,
                     checkpoint_path='checkpoint_best.pt', result_path='results.json',
                     patience=5, temperature=1.0,
                     soft_weight=0.5, hard_weight=0.5):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model.to(device), teacher_model.eval()
    student_model.to(device)

    train_losses, val_losses = [], []  # Fix variable name
    train_metrics_list, val_metrics_list = [], []
    best_f1_score = float(-1)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    for epoch in range(epochs):
        if pred_distill:
            student_model.count_epochs()
        # Loss and Metrics
        print("Training")
        train_loss, train_metrics = step(student_model, teacher_model, ver, train_dataloader, optimizer, device, temperature, soft_weight, hard_weight, pred_distill, max_grad_norm = 1.0, mode='Train')
        print("Validating")
        val_loss, val_metrics = step(student_model, teacher_model, ver, val_dataloader, optimizer, device, temperature, soft_weight, hard_weight, pred_distill, max_grad_norm = 1.0, mode='Val')
        print('Testing')
        results, y_pred, y_true = evaluate_model(student_model, ver, test_dataloader, average = 'macro')

        # Obtain Loss and Metrics
        # Obtain Loss and Metrics
        train_losses.append(train_loss)
        train_metrics_list.append(train_metrics)
        val_losses.append(val_loss)
        val_metrics_list.append(val_metrics)

        # Step the scheduler
        # scheduler.step(val_loss)
        scheduler.step()

        # Early Stopping
        early_stopping(val_loss, student_model, optimizer, scheduler)  # Fix variable name

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
                'model_state_dict': student_model.state_dict(),  # Fix variable name
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
