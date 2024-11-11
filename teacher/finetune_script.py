

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch
import argparse
import os

# Move to parent folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dataset import create_dataset
from teacher.finetune_function import finetune_teacher

# Set manual seed
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

folder_path = 'teacher/checkpoints'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            lr_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_factor = max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        # Print the current step and learning rate factor
        print(f"Step: {current_step}, Learning Rate Factor: {lr_factor}")

        return lr_factor
    return LambdaLR(optimizer, lr_lambda)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ver', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--freeze_layers', type=str2bool, default=True)
    parser.add_argument('--gradual_unfreezing', type=str2bool, default=True)
    parser.add_argument('--num_epochs_freeze', type=int, default=2)
    parser.add_argument('--unfreeze_steps', type=int, default=1)
    args = parser.parse_args()

    # Dataset, Dataloader
    train_dataset, val_dataset, test_dataset = create_dataset(data_path = args.data_path,
                                                              batch_size = args.batch_size)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Check whether English version is used correctly
    sample = next(iter(train_dataloader))
    print('Sample Shape: ', sample['input_ids'].shape)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(tokenizer.decode(sample['input_ids'][0], skip_special_tokens = True))

    # Import Model
    print(f'Freeze Layer is: {args.freeze_layers}')
    if args.ver == 1:
        from models.roberta_model import ASBA_PhoBertCustomModel
        model = ASBA_PhoBertCustomModel(roberta_version = args.model_name,
                                        num_labels = train_dataset.num_labels(),
                                        num_epochs_freeze = args.num_epochs_freeze,
                                        unfreeze_steps = args.unfreeze_steps,
                                        freeze_layers = args.freeze_layers,)
        criterion = nn.CrossEntropyLoss()
        print(f'Implement CrossEntropyLoss')

    else :
        from models.roberta_model_ver2 import ASBA_PhoBertCustomModel, VLSP2018MultiTask_Huy
        # model = ASBA_PhoBertCustomModel(roberta_version = args.model_name,
        #                                 num_labels = train_dataset.num_labels(),
        #                                 loss = 'bce-logits',
        #                                 num_epochs_freeze = args.num_epochs_freeze,
        #                                 unfreeze_steps = args.unfreeze_steps,
        #                                 freeze_layers = args.freeze_layers,)
        model = VLSP2018MultiTask_Huy(roberta_version = args.model_name, 
                                      num_labels = train_dataset.num_labels(), 
                                      num_epochs_freeze = args.num_epochs_freeze,
                                      unfreeze_steps = args.unfreeze_steps,
                                      freeze_layers = args.freeze_layers,)
        criterion = nn.BCEWithLogitsLoss()
        print(f'Implement BCEWIithLogitsLoss')

    # Finetuning Config

    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-04, betas=(0.9, 0.98), weight_decay=1e-3)

    # Scheduler with warm-up and linear decay
    num_training_steps = args.epochs * len(train_dataloader)
    # num_warmup_steps = args.num_epochs_freeze * len(train_dataloader)
    num_warmup_steps = 500
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Finetuning
    save_name = args.model_name.split('/')[-1]
    train_loss, test_loss, train_metrics, test_metrics = finetune_teacher(model=model,
                                                                          ver = args.ver,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          scheduler=scheduler,
                                                                          checkpoint_path = f'teacher/checkpoints/{save_name}-Teacher-ver{args.ver}-best.pt',
                                                                          result_path = f'teacher/checkpoints/{save_name}-Teacher-ver{args.ver}-results.json',
                                                                          max_grad_norm=1.0,
                                                                          epochs=args.epochs,
                                                                          patience=5,
                                                                          gradual_unfreezing = args.gradual_unfreezing)

    results = {
        'Train': {
            'Loss': train_loss,
            'Metrics': train_metrics,
        },
        'Val': {
            'Loss': test_loss,
            'Metrics': test_metrics,
        }
    }

    # # Save results to a JSON file
    # import json
    # id = args.model_name.split('/')[-1]
    # with open(f'{id}_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
