from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from typing import cast, List
from langchain_core.documents import Document
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from utilities import load_document, split_into_chunks, add_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
model = ChatGroq(model="Llama3-8b-8192")

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin! \n  Allowed formats: `.txt`, `.pdf`, `.docx`",
            accept=["text/plain", ".pdf", ".docx"],
            max_size_mb=20,
            timeout=180,
            max_files=5,
        ).send()

    all_document_chunks = []
    for file_obj in files:
        msg = cl.Message(content=f"Processing `{file_obj.name}`...")
        await msg.send()
        documents = load_document(file_obj)
        document_chunks = split_into_chunks(documents)
        all_document_chunks.extend(document_chunks)

    vector_database = await cl.make_async(Chroma.from_documents)(all_document_chunks, embeddings)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        model,
        chain_type="stuff",
        retriever=vector_database.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()
    cl.user_session.set("chain", chain)  # Changed from 'runnable' to 'chain'


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")  # Get the chain we stored
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall({"question": message.content}, callbacks=[cb])  # Pass question as a dict
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content, name=source_name, display="side"
                )
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()