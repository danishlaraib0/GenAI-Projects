import streamlit as st
import os
from langchain import HuggingFaceHub
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
import cassio
from langchain.text_splitter import CharacterTextSplitter


#Loading the Model
llm=HuggingFaceHub(huggingfacehub_api_token="Put_your_apikey_here_fromHugginface", repo_id="deepseek-ai/DeepSeek-V3",model_kwargs={"temperature":0.6,"max_length":500})


#astradb credentials (Get your credentials from datastax (like i used it from there. these keys are just placeholders here))
ASTRA_DB_APPLICATION_TOKEN="AstraCS:idpnjzcvdsaMIKvzRd:89e02db55d0guavbhdzvueabhdzivubfdcbzjv1a7b7a9fa1ac21"
ASTRA_DB_ID="103scnj-2vsdn-43vdj-vf05-avsdsldm501ee1"
#building connection to astradb
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)
    
# Designing the tokenizer (We are using BERT Tokenizer Here)
from transformers import BertTokenizer, BertModel
import torch

class CustomEmbedding:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def embed_query(self, text):
        """
        Generate an embedding for a single query text.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        # Generate embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(**inputs)

        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents.
        """
        embeddings = []
        for text in texts:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

            # Generate embeddings
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model(**inputs)

            # Use the [CLS] token embedding as the sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)

        return embeddings
    
embedding_model = CustomEmbedding()
astra_vector_store2=Cassandra(embedding=embedding_model,table_name='test3',session=None,keyspace=None)


# Streamlit app
st.title("File Upload and Query App")

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
if uploaded_file is not None:
    try:
        pdfreader = PdfReader(uploaded_file)
        raw_text=''
        for i,page in enumerate(pdfreader.pages):
            content=page.extract_text()
            if content:
                raw_text+=content
    
        text_splitter=CharacterTextSplitter(separator = "\n",chunk_size = 800,chunk_overlap = 200,length_function = len,)
        texts = text_splitter.split_text(raw_text)
        ids = astra_vector_store2.add_texts(texts)
        astra_vector_index2 = VectorStoreIndexWrapper(vectorstore=astra_vector_store2)
        st.success("File uploaded and embeddings stored!")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Query input
query_text = st.text_input("Enter your query:")
if query_text:
    answer = astra_vector_index2.query(query_text, llm=llm).strip()
    st.write("Answer:", answer)