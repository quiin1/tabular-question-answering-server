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

data = [["Party", "Votes", "%", "Seats", "+/\u2013"], 
     ["Cameroonian National Union", "3,293,428", "100", "120", "+70"], 
     ["Invalid/blank votes", "1,577", "\u2013", "\u2013", "\u2013"], 
     ["Total", "3,295,005", "100", "120", "+70"], 
     ["Registered voters/turnout", "3,348,989", "98.4", "\u2013", "\u2013"], 
     ["Source: Nohlen et al.", "Source: Nohlen et al.", "Source: Nohlen et al.", "Source: Nohlen et al.", "Source: Nohlen et al."]]

@app.post("/predict")
def predict(data: dict):
    table = pd.DataFrame.from_records(data["table"][1:], columns=data["table"][0])
    
    query = data["query"]

    encoding = tokenizer(table=table, query=query, return_tensors="pt")

    output = model.generate(**encoding)

    return (tokenizer.batch_decode(output, skip_special_tokens=True))