from fastapi import FastAPI ,Form ,Request, Response, File,Depends,HTTPException,status
from fastapi.responses import  RedirectResponse, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn 
import os
import json
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from src.helper import load_pdf_files,fetch_page_content,text_split,load_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from src.prompt import system_prompt
from pydantic import BaseModel

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embedding = load_embedding_model()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever =  docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0
   
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
reg_chain = create_retrieval_chain(retriever,question_answer_chain)




app = FastAPI()
app.mount("/static",StaticFiles(directory="static"),name="static")

template = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request : Request):
    return template.TemplateResponse("index.html",{"request" : request})

class ChatRequest(BaseModel):
    msg: str

@app.post("/chat")
async def chat(payload: ChatRequest):
    response = reg_chain.invoke({"input": payload.msg})
    return {"response": response['answer']}



if __name__ == "__main__":
    uvicorn.run("app:app",host='0.0.0.0',port=8080,reload=True)


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React origin
    allow_methods=["*"],
    allow_headers=["*"],
)