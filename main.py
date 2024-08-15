from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

llm = ChatOllama(model=os.getenv('OLLAMA_MODEL'), temperature=0, base_url=os.getenv('OLLAMA_BASE_URL'))

prompt = PromptTemplate.from_template(
    """Analisa teks percakapan berikut ke dalam format di bawah serta klasifikasikanlah ke dalam salah satu label:

    Konteks Percakapan: 
    {{konteks_percakapan}}

    Keyword Extraction: 
    {{keyword_extraction}}

    Intent Recognition: 
    {{intent_recognition}}

    Kualifikasi Lead: 
    {{kualifikasi_lead}}

    Label: [L1-Qualified, S_JUNK, Qualified, L1-Junk]

    {input}
    """
)

output_parser = StrOutputParser()

class InputData(BaseModel):
    input: List[Dict[str, str]]

@app.post("/generate/")
async def generate_response(data: InputData):
    try:
        formatted_input = "\n".join([f'{key}: {value}' for message in data.input for key, value in message.items()])
        
        chain = prompt | llm | output_parser
        response = chain.invoke({"input": formatted_input})
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
