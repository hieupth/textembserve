import os
import ovmsclient
import numpy as np
from typing import Union
from fastapi import FastAPI
from hftokenizer import BertTokenizer, PhobertTokenizer

# Read tokenizer name.
TOKENIZER = os.getenv("TOKENIZER", "bert")
# Is words segment?
IS_WORD_SEGMENT = os.getenv("IS_WORD_SEGMENT", False)
# Model path, default is /tokenizer
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME", "/tokenizer")

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
client = ovmsclient.make_grpc_client(os.getenv("OVMS_URI", "ovms:9000"))

# Make fastapi
app = FastAPI()

@app.post("/")
async def predict(mess: str):
    tokenized = tokenizer(mess, return_tenros="np")
    inputs = {
        "input_ids": np.array(tokenized["input_ids"], dtype=np.int64),
        "token_type_ids": np.array(tokenized["token_type_ids"], dtype=np.int64),
        "attention_mask": np.array(tokenized["attention_mask"], dtype=np.int64),
    }
    return client.predict(model_name="bk", inputs=inputs)
