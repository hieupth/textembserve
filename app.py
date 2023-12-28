import os
import json
import torch
import ovmsclient
import numpy as np
from os import environ
from fastapi import FastAPI
from hftokenizer import BertTokenizer, PhobertTokenizer

environ["GRPC_DNS_RESOLVER"] = "native"
# Read tokenizer name.
TOKENIZER = os.getenv("TOKENIZER", "phobert")
# Is words segment?
IS_WORD_SEGMENT = os.getenv("IS_WORD_SEGMENT", True)
# Model path, default is /tokenizer
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME", "./tokenizer")

# Create bert tokenizer.
if TOKENIZER.lower() == "bert":
  tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, os.path.join(MODEL_NAME_OR_PATH, "vocab.txt")
  )
# Create phobert tokenizer.
elif TOKENIZER.lower() == "phobert":
  tokenizer = PhobertTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    vocab_file=os.path.join(MODEL_NAME_OR_PATH, "vocab.txt"),
    merges_file=os.path.join(MODEL_NAME_OR_PATH, "bpe.codes"),
    words_segment=IS_WORD_SEGMENT
  )

# Make openVINO client
client = ovmsclient.make_grpc_client(os.getenv("OVMS_URI", "localhost:9000"))

# Make fastapi
app = FastAPI()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@app.post("/")
async def predict(data: dict):
  tokenized = tokenizer(data["message"])
  metadata = client.get_model_metadata(data["model"])
  inputs = dict()
  for k in metadata["inputs"].keys():
    inputs.update({k: np.array([tokenized[k]], dtype=np.int64)})
  res = client.predict(model_name=data["model"], inputs=inputs)
  res = mean_pooling(
    torch.from_numpy(np.asarray(res)), 
    torch.from_numpy(np.asarray(tokenized["attention_mask"]))
  )
  return json.dumps(res.squeeze().tolist())