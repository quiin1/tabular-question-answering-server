from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    BartForConditionalGeneration,
    TapexTokenizer,
)
import pandas as pd

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

tokenizer = TapexTokenizer.from_pretrained('model')
model = BartForConditionalGeneration.from_pretrained('model')


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}

@app.post("/predict")
async def predict(data: dict):
    table = pd.DataFrame.from_records(data["table"][1:], columns=data["table"][0])
    
    query = data["query"].lower()

    encoding = tokenizer(table=table, query=query, return_tensors="pt")
    
    output = model.generate(**encoding)

    return (tokenizer.batch_decode(output, skip_special_tokens=True))