from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import psycopg2
import uuid

load_dotenv()

app = FastAPI()

llm = ChatOllama(model=os.getenv('OLLAMA_MODEL'), temperature=0, base_url=os.getenv('OLLAMA_BASE_URL'))

classification_prompt = PromptTemplate.from_template(
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

chat_prompt = PromptTemplate.from_template(
    """Chat dengan konteks berikut:

    {input}
    """
)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        database=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        thread_id UUID,
                        message TEXT
                    )''')
    conn.commit()
    cursor.close()
    conn.close()

init_db()

class ChatInput(BaseModel):
    thread_id: Optional[str] = None
    message: str

@app.post("/chat/")
async def chat(data: ChatInput):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if data.thread_id:
            thread_id = data.thread_id
            cursor.execute("SELECT message_text FROM messages WHERE thread_id = %s", (thread_id,))
            messages = cursor.fetchall()
            context = "\n".join([msg[0] for msg in messages])
        else:
            thread_id = str(uuid.uuid4())
            context = ""

        formatted_input = f"{context}\n{data.message}" if context else data.message
        chain = chat_prompt | llm | output_parser
        response = chain.invoke({"input": formatted_input})

        cursor.execute("INSERT INTO chat_history (thread_id, message) VALUES (%s, %s)", (thread_id, data.message))
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id = %s", (thread_id,))
        message_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        if message_count >= 4:
            classification_result = await classify(thread_id)
        else:
            classification_result = None

        return {
            "thread_id": thread_id, 
            "response": response, 
            "classification": classification_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def classify(thread_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT message_text FROM messages WHERE thread_id = %s", (thread_id,))
        messages = cursor.fetchall()
        context = "\n".join([msg[0] for msg in messages])
        cursor.close()
        conn.close()

        formatted_input = context
        chain = classification_prompt | llm | output_parser
        response = chain.invoke({"input": formatted_input})
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
