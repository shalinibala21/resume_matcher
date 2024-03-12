import openai
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.llms import OpenAI
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
import os
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":"","type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    logging.info(f'printing embeddings')
    logging.info(embeddings)

    return embeddings





# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary

# Function to push data to Vector Store - Pinecone
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    from pinecone import Pinecone
    os.environ['PINECONE_API_KEY'] = ''
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    pc = Pinecone(api_key=api_key)

    #from pinecone import ServerlessSpec, PodSpec
    #spec = PodSpec(environment=environment)

    index_name='test'
    index = pc.Index(index_name)
    logging.info(index.describe_index_stats())
    


    #index.upsert(docs,  namespace = "example_namespace")
    #logging.info(index.describe_index_stats())

    from langchain.vectorstores import Pinecone as PineconeVectorStore
    PineconeVectorStore.from_documents(docs, embeddings, index_name='test')

    index_name='test'
    index = pc.Index(index_name)
    logging.info(index.describe_index_stats())
    





# Function to get relevant documents from Vector Store - Pinecone based on user input
def similar_docs(query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, unique_id):
    
    from pinecone import Pinecone
    os.environ['PINECONE_API_KEY'] = ''
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    pc = Pinecone(api_key=api_key)

    #insert index name
    index_name='test'
    index = pc.Index(index_name)
    logging.info(index.describe_index_stats())

    text_field = "text"

    from langchain.vectorstores import Pinecone
    vectorstore = Pinecone(index, embeddings, text_field)




    similar_docs=vectorstore.similarity_search_with_relevance_scores(query, int(k) )
    #similar_docs = vectorstore.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    logging.info('*****SIMILAR DOCS*****')
    logging.info(similar_docs)



    return similar_docs