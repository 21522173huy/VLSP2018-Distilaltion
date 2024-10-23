
from transformers import AutoModel
from torch import nn
import torch 

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, 4)

    def forward(self, features, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ASBA_T5CustomModel(nn.Module):
  def __init__(self, t5_version, num_labels, freeze_encoder = False):
    super().__init__()
    base = AutoModel.from_pretrained(t5_version)
    self.hidden_size = base.encoder.config.hidden_size
    self.num_labels = num_labels
    # Encoder
    self.encoder = base.encoder

    # Classifier
    self.classifier = ClassifierLayer(self.hidden_size, num_labels)
    if freeze_encoder:
      self.freeze_encoder_fn()

  def freeze_encoder_fn(self):
    for param in self.encoder.parameters():
        param.requires_grad = False

    # Ensure the classifier layer's parameters are not frozen
    for param in self.classifier.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    output = self.encoder(input_ids = input_ids,
                          attention_mask = attention_mask).last_hidden_state
    
    cls_token = output[:, 0, :] # Taking the [CLS] token
    expanded_cls_token = cls_token.unsqueeze(1).expand(-1, self.num_labels, -1)
    
    logits = self.classifier(expanded_cls_token)

    return logits
