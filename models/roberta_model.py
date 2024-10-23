import torch
from transformers import AutoModel
from torch import nn

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, 4)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ASBA_PhoBertCustomModel(nn.Module):
    def __init__(self, roberta_version: str, num_labels: int, num_epochs_freeze=2, unfreeze_steps=1):
        super(ASBA_PhoBertCustomModel, self).__init__()
        base = AutoModel.from_pretrained(roberta_version)
        self.hidden_size = base.config.hidden_size

        # Embedding
        self.embeddings = base.embeddings

        # BLocks
        self.encoder = base.encoder

        # Classifier
        self.classifier = ClassifierLayer(self.hidden_size)

        # Other unfreeze_steps
        self.set_up_other_components(num_labels, num_epochs_freeze, unfreeze_steps)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(f"Using device: {self.device}")

        print(f'Train Only Classifier Layer')
        self.set_up_layer()

    def set_up_layer(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def set_up_other_components(self, num_labels, num_epochs_freeze, unfreeze_steps):
        self.num_labels = num_labels
        self.num_epochs_freeze = num_epochs_freeze
        self.unfreeze_steps = unfreeze_steps
        self.current_epoch = 0
        self.current_unfreeze_step = 0

    def gradual_unfreeze(self):
        total_layers = len(self.encoder.layer)
        
        # Unfreeze embeddings first
        if self.current_unfreeze_step == 0:
            for param in self.embeddings.parameters():
                param.requires_grad = True
            print("Unfroze embeddings")
        else:
            layers_to_unfreeze = self.current_unfreeze_step * self.unfreeze_steps
            layers_to_unfreeze = min(layers_to_unfreeze, total_layers)  # Ensure we don't exceed total layers
            for i in range(layers_to_unfreeze):
                for param in self.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"Unfroze {layers_to_unfreeze} layers")

        self.current_unfreeze_step += 1

    def count_epochs(self):
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

        cls_token = encoder_output[:, 0, :] # Taking the [CLS] token
        expanded_cls_token = cls_token.unsqueeze(1).expand(-1, self.num_labels, -1)
        return self.classifier(expanded_cls_token)
