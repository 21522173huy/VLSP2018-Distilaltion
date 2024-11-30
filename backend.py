from models.roberta_model_ver2  import VLSP2018MultiTask_Huy
from models.roberta_model import ASBA_PhoBertCustomModel
import torch
from transformers import AutoTokenizer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from dataset.dataset import create_dataset

# Load VnCoreNLP
import os
os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk-11.0.2'

import py_vncorenlp
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:\\Users\\tuleh\\Desktop\\VLSP2018-Distillation\\VnCoreNLP')
os.chdir('C:\\Users\\tuleh\\Desktop\\VLSP2018-Distillation') # Change back to VLSP2018-Distilaltion directory

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

app = FastAPI()

class TextInput(BaseModel):
    text: str
    dataset: str
    id_option: str

def load_checkpoint(model, checkpoint):
  if len(checkpoint.keys()) == 3:
    model.load_state_dict(checkpoint['model_state_dict'])
  else:
    model.load_state_dict(checkpoint)

def load_model(dataset, id_option, device):
    MODEL_ID = {
        'PhoBertLarge': 'vinai/phobert-large',
        # 'PhoBertBase': 'vinai/phobert-base-v2'
    }
    model_id = MODEL_ID[id_option]
    if dataset == 'Restaurant':
        best_model = VLSP2018MultiTask_Huy(roberta_version=model_id,
                                        num_labels=34 if dataset == 'Hotel' else 12).to(device)
    else:
        best_model = ASBA_PhoBertCustomModel(roberta_version=model_id,
                                            num_labels=34 if dataset == 'Hotel' else 12)
    best_checkpoint = torch.load(f"teacher/checkpoints/{dataset}/{id_option}/Best.pt", map_location=torch.device(device))
    load_checkpoint(best_model, best_checkpoint)
    best_model.eval()
    return best_model

def analyzing_aspect_sentiment(model, tokens, idx2polarity_fn, device):
    input_ids, attention_mask = tokens['input_ids'].to(device), tokens['attention_mask'].float().to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        if len(output.shape) == 3: # batch_size, num_labels, 4
            pred = output.argmax(dim=-1) # batch_size, num_labels

        else : # batch_size, num_labels * 4
            reshaped_output = output.view(-1, model.num_labels, 4)
            pred = reshaped_output.argmax(dim=-1) # batch_size, num_labels
    
    aspect_polarity_pair = idx2polarity_fn(pred.squeeze(dim=0))

    return aspect_polarity_pair

global models, device
device = "cuda" if torch.cuda.is_available() else "cpu"
models = {} # Global variables to store the models

@app.on_event("startup")
async def startup_event():
    # Load models for both datasets and options
    for dataset in ["Hotel"]:
        for id_option in ["PhoBertLarge"]:
            models[(dataset, id_option)] = load_model(dataset, id_option, device)

@app.post("/analyze")
async def analyze_text(input: TextInput):
    # Input
    text = input.text
    dataset = input.dataset
    id_option = input.id_option

    # Load model
    model = models[(dataset, id_option)]

    # Word Segmentation and Tokenization
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')

    # segmented_text 
    segmented_text = rdrsegmenter.word_segment(text)[0]
    tokens = tokenizer(segmented_text, return_tensors="pt")

    # Aspect-Polarity Mapping
    data_path = f'dataset/preprocessed_{dataset.lower()}'
    _, _, processed_test_dataset = create_dataset(data_path = data_path)
    idx2polarity_fn = processed_test_dataset.idx2polarity

    # Analyzing Aspect Sentiment
    sentiments = analyzing_aspect_sentiment(model, tokens, idx2polarity_fn = idx2polarity_fn, device = device)

    return sentiments

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
