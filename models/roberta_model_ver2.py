
import torch
from transformers import AutoModel, AutoConfig
from torch import nn

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size, num_labels = None):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        if num_labels == None:
          self.out_proj = nn.Linear(hidden_size, 4)
        else : self.out_proj = nn.Linear(hidden_size, num_labels * 4)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ASBA_PhoBertCustomModel(nn.Module):
    def __init__(self, roberta_version: str, num_labels: int, num_epochs_freeze=2, unfreeze_steps=1, loss = 'cross-entropy', freeze_layers=True):
        super(ASBA_PhoBertCustomModel, self).__init__()
        base = AutoModel.from_pretrained(roberta_version)
        self.hidden_size = base.config.hidden_size

        # Embedding
        self.embeddings = base.embeddings

        # BLocks
        self.encoder = base.encoder

        # Classifier
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                ClassifierLayer(self.hidden_size)
            )
            for _ in range(num_labels)
        ])

        # Other unfreeze_steps
        self.set_up_other_components(num_labels, num_epochs_freeze, unfreeze_steps, loss, freeze_layers)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(f"Using device: {self.device}")

        if self.freeze_layers:
            self.set_up_layer()

    def set_up_layer(self):
        print(f'Train Only Classifier Layer')
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifiers.parameters():
            param.requires_grad = True

    def set_up_other_components(self, num_labels, num_epochs_freeze, unfreeze_steps, loss, freeze_layers):
        self.loss = loss
        self.num_labels = num_labels
        self.num_epochs_freeze = num_epochs_freeze
        self.unfreeze_steps = unfreeze_steps
        self.current_epoch = 0
        self.current_unfreeze_step = 1
        self.all_layers_unfrozen = False # Flag to check if all layers are unfrozen
        self.freeze_layers = freeze_layers

    def gradual_unfreeze(self):
        if self.all_layers_unfrozen:
            return

        total_layers = len(self.encoder.layer)

        start_layer = total_layers - (self.current_unfreeze_step * self.unfreeze_steps)
        end_layer = total_layers - ((self.current_unfreeze_step - 1) * self.unfreeze_steps)
        start_layer = max(start_layer, 0)  # Ensure we don't go below 0

        for i in range(start_layer, end_layer):
            for param in self.encoder.layer[i].parameters():
                param.requires_grad = True
        print(f"Unfroze layers {start_layer} to {end_layer - 1}")

        self.current_unfreeze_step += 1

        # Check if all layers are unfrozen
        if start_layer == 0:
            self.all_layers_unfrozen = True

    def count_epochs(self):
        if self.freeze_layers == False : return
        if self.current_epoch >= self.num_epochs_freeze:
            print(f'Unfreezing layers gradually')
            self.gradual_unfreeze()
        self.current_epoch += 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Embedding
        embedding_output = self.embeddings(input_ids)
        if attention_mask is not None:
            # Reshape the attention mask to match the shape required for multi-head attention
            attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_length)

        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask)[0]

        cls_token = encoder_output[:, 0, :] # Taking the [CLS] token, (batch_size, 768)
        logits = [classifier(cls_token) for classifier in self.classifiers] # list of 34 item (batch_size, 4)
        logits = torch.stack(logits, dim=1) # (batch_size, 34, 4)
        if self.loss == 'cross-entropy':
          return logits # (batch_size, 34, 4)
        else :
          return logits.view(-1, self.num_labels * 4) # (batch_size, 34 * 4)

class VLSP2018MultiTask_Huy(nn.Module):
    def __init__(self, roberta_version, num_labels, num_layers = None, num_epochs_freeze = 2, unfreeze_steps = 1, freeze_layers = True):
        super(VLSP2018MultiTask_Huy, self).__init__()

        self.num_labels = num_labels
        if num_layers is not None: # For Student Implementation
            config = AutoConfig.from_pretrained(roberta_version)
            config.num_hidden_layers = num_layers
            self.pretrained_bert = AutoModel.from_config(config)
            self.pretrained_bert.init_weights()
        
        else:
            self.pretrained_bert = AutoModel.from_pretrained(roberta_version, output_hidden_states=True)
            
        self.hidden_size = self.pretrained_bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size * 4, 4)
            for _ in range(self.num_labels)
        ])
        self.flatten_onehot_labels = nn.Linear(4 * self.num_labels, 4 * self.num_labels)

        # Other unfreeze_steps
        self.set_up_other_components(num_epochs_freeze, unfreeze_steps, freeze_layers)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(f"Using device: {self.device}")

        if self.freeze_layers: # Freeze Layer or Not
            self.set_up_layer()

    def set_up_layer(self):
        print(f'Train Only Classifier Layer')
        for param in self.pretrained_bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.pretrained_bert.encoder.parameters():
            param.requires_grad = False
        for param in self.classifiers.parameters():
            param.requires_grad = True
        for param in self.flatten_onehot_labels.parameters():
            param.requires_grad = True

    def set_up_other_components(self, num_epochs_freeze, unfreeze_steps, freeze_layers):
        self.num_epochs_freeze = num_epochs_freeze
        self.unfreeze_steps = unfreeze_steps
        self.current_epoch = 0
        self.current_unfreeze_step = 1
        self.all_layers_unfrozen = False  # Flag to check if all layers are unfrozen
        self.freeze_layers = freeze_layers

    def gradual_unfreeze(self):
        if self.all_layers_unfrozen:
            return

        total_layers = len(self.pretrained_bert.encoder.layer)

        start_layer = total_layers - (self.current_unfreeze_step * self.unfreeze_steps)
        end_layer = total_layers - ((self.current_unfreeze_step - 1) * self.unfreeze_steps)
        start_layer = max(start_layer, 0)  # Ensure we don't go below 0

        for i in range(start_layer, end_layer):
            for param in self.pretrained_bert.encoder.layer[i].parameters():
                param.requires_grad = True
        print(f"Unfroze layers {start_layer} to {end_layer - 1}")

        self.current_unfreeze_step += 1

        # Check if all layers are unfrozen
        if start_layer == 0:
            self.all_layers_unfrozen = True

    def count_epochs(self):
        if not self.freeze_layers:
            return
        if self.current_epoch >= self.num_epochs_freeze:
            print(f'Unfreezing layers gradually')
            self.gradual_unfreeze()
        self.current_epoch += 1

    def forward(self, input_ids, attention_mask=None):
        outputs = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        
        hidden_states = outputs.hidden_states  # All hidden states
        attentions = outputs.attentions  # All attention scores
        
        pooled_output = torch.cat(hidden_states[-4:], dim=-1)[:, 0, :]  # Concatenate last 4 hidden states and take the [CLS] token
        x = self.dropout(pooled_output)

        classifier_outputs = [classifier(x) for classifier in self.classifiers]
        flattened_output = self.flatten_onehot_labels(torch.cat(classifier_outputs, dim=-1))
        
        return flattened_output, attentions, hidden_states

