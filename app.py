import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and when asked about a particular point please ensure that you summarize all the content related to it in an elaborated format considering all the information in the context and summarize it. Make sure to provide all the details and explain it in a very comprehensive manner. If the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def ask_question(pdf_file, question):
    text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
    print("Response:", response)  # Print out the response for debugging
    
    return response['output_text'] if 'output_text' in response else "Error: No answer found"

def pdf_query(pdf_file, question):
    answer = ask_question(pdf_file, question)
    return str(answer)

file_input = gr.File(label="Upload PDF")
question_input = gr.Textbox(label="Enter Question")
output_text = gr.Textbox(label="Answer")

iface = gr.Interface(fn=pdf_query, inputs=[file_input, question_input], outputs=output_text, title="PDF Query Tool", description="Ask questions about a PDF document and get answers.")
iface.launch(debug=True)
