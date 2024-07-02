import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

@cl.on_chat_start
async def start():
    logging.info("Chat session started")
    try:
        chain = qa_bot()
        cl.user_session.set("chain", chain)
        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to Medical Bot. What is your query?"
        await msg.update()
    except Exception as e:
        logging.error("Error initializing chat: %s", e)
        await cl.Message(content="Error initializing chat. Please try again.").send()

@cl.on_message
async def main(message: cl.Message):
    logging.info("Received message: %s", message.content)
    try:
        chain = cl.user_session.get("chain")
        if chain is None:
            logging.error("Chain not found in user session")
            await cl.Message(content="Error: Chain not found").send()
            return

        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        res = await chain.ainvoke(message.content, callbacks=[cb])  # Use ainvoke instead of acall
        logging.info("Response received: %s", res)
        answer = res["result"]
        sources = res["source_documents"]

        if sources:
            answer += f"\nSources: {sources}"
        else:
            answer += "\nNo sources found"

        await cl.Message(content=answer).send()
    except Exception as e:
        logging.error("Error processing message: %s", e)
        await cl.Message(content="Error processing your query. Please try again.").send()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cl.run(__file__, ['-w'])  # Run ChainLit with debug mode
