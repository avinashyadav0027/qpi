import torch
import time
import requests
import torch.nn as nn
import torch.nn.functional as F
from model.bert import BERT
from transformers import BertModel
from transformers import BertTokenizer

# from transformers.utils import http_backoff
# http_backoff.verify_ssl = False

import requests

class NoVerifySession(requests.Session):
    def __init__(self, *args, **kwargs):
        super(NoVerifySession, self).__init__(*args, **kwargs)
        self.verify = False

requests.Session = NoVerifySession


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
teacher_model = BertModel.from_pretrained("bert-base-uncased")

file_path = 'wiki_97'

teacher_output_dict = {}


def generate_distill_data():
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for batch_no, line in enumerate(lines):
    
        inputs = tokenizer(line.strip(), return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None) 
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, token_type_ids, attention_mask)
            teacher_embeddings = teacher_outputs.last_hidden_state  
            teacher_output_dict[batch_no] = teacher_embeddings


def evaluate_fitness(individual, gpu_ram, expected_latency):
    
    student_model = BERT(hidden=individual[0], n_layers=individual[1])

    loss_fn = torch.nn.MSELoss()
    
    model_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

# Calculate size in bytes
    model_size_bytes = model_size * 4  # for 32-bit float (4 bytes)

    # Convert bytes to gigabytes
    model_size_gb = model_size_bytes / (1024 ** 3)  # converting bytes to gigabytes

    if model_size_gb > gpu_ram:
        return float('-inf'),  # Return negative infinity if model size exceeds GPU RAM

    # Read input lines from the file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    total_batches = 0
    total_latency = 0
    total_loss = 0
    # Loop through batches
    for batch_no, line in enumerate(lines):
        # Tokenize the input
        inputs = tokenizer(line.strip(), return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None) 
        attention_mask = inputs['attention_mask']

        # Measure latency
        start_time = time.time()
        student_output = student_model(input_ids, segment_info=token_type_ids, attention_mask=attention_mask)
        latency = time.time() - start_time
        total_latency += latency

        # Fetch teacher's output for distillation
        teacher_output = teacher_output_dict[batch_no]
        
        # Calculate loss (distillation loss) between student and teacher
        loss = loss_fn(student_output, teacher_output)
        total_loss += loss.item()
        total_batches += 1

    # Calculate average latency
    avg_latency = total_latency / total_batches if total_batches > 0 else float('inf')

    # Penalize the loss if the latency exceeds the expected threshold
    latency_penalty = max(0, avg_latency - expected_latency)
    penalized_loss = total_loss + latency_penalty

    return penalized_loss,
