from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from .utils import json_to_sql, create_sql_agent, delete_all_tables

from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.chains.llm import LLMChain

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

con = sqlite3.connect('Table.db')
llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')

@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}

@app.post("/predict")
async def predict(data: dict):
    if 'table' in data:
        json_to_sql(con, data)
    else:
        delete_all_tables(con)
        cur = con.cursor()
        cur.executescript(data['sql'])

    db = SQLDatabase.from_uri("sqlite:///./Table.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )

    result = agent_executor(data['query'])
    
    return {
        'sql': result['intermediate_steps'][-1][0].tool_input,
        'output': result['intermediate_steps'][-1][1],
        'answer': result['output']
    }