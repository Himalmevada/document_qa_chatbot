from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from typing import cast
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def split_into_chunks():
    pass

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload a text file to begin! \n  Allowed formats: `.txt`, `.pdf`, `.docx`",
            accept=["text/plain",".pdf", ".docx"],
            max_size_mb=20,
            timeout=180,
            max_files=5
        ).send()

    print(files)

    msg = cl.Message(content="Processing Files...")
    await msg.send()

    model = ChatGroq(model="Llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable ai assistant who provides accurate and eloquent answers to the use's question.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
    ):
        await msg.stream_token(chunk)

    await msg.send()