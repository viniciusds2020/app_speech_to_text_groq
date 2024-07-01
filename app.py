import whisper
import torch
import streamlit as st
from st_audiorec import st_audiorec
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Model Llama3
llm = ChatGroq(api_key=GROQ_API_KEY, model='llama3-70b-8192', temperature=0.7)

def load_web_page(page_url):
    loader = WebBaseLoader(page_url)
    data = loader.load()
    return data

page_url = 'https://www.hostinger.com/tutorials/how-to-start-a-blog?utm_campaign=Generic-Tutorials-DSA|NT:Se|LO:BR-EN&utm_medium=ppc&gad_source=1&gclid=CjwKCAjwr7ayBhAPEiwA6EIGxGcwREvm1aYAN4S6Uj18NX_Yu04qmp52K7EhGuTexjYJjWgY51daaBoC5GoQAvD_BwE'
docs = load_web_page(page_url)

# Documentos
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False)

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts

data = split_documents(docs)

# Embeddings - Hugging Face - AllMiniLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'},
                                    encode_kwargs = {'normalize_embeddings': False})

# Instanciando o banco de dados de vetores - ChromaDB
dbqa = Chroma.from_documents(data,embeddings)

# This creates the recording widget
wav_audio_data = st_audiorec()

# Interface Streamlit
st.title("Assistente Virtual")

# After we stop the recording we will have some audio/WAV data available
if wav_audio_data is not None:

    # This allows us to listen to the recording using the built in streamlit widget
    st.audio(wav_audio_data, format="audio/wav")

    # The easiest way to load the data into whisper in the correct format
    # is to just store them in a temporary file.
    with open("audio.wav", "wb") as f:
        f.write(wav_audio_data)

    # Whisper can deal with all the problems around processing wav files adequately.
    # On the plus side this is actually quite fast since when loading from a file we can use torch directly to convert
    data = whisper.load_audio("/content/audio.wav")

    # We can load one of the models here, the base model requires fewest amount of memory
    model = whisper.load_model("small")

    # This is the high level way of transcribing your audio into text.
    result = model.transcribe(data)

    # Result is a dictionary containing "text" and "segments" and "language"
    st.write(result["text"])

    # There is also a lower level way that allows you to tweak your audio sample
    audio = whisper.pad_or_trim(torch.from_numpy(data).float())
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    # This allows us to check for other potential candidates,
    # for example if we want to do some debugging or calculating some metrics
    st.write(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    st.write(result.text)

    # Q&A chain
    pdf_qa = ConversationalRetrievalChain.from_llm(llm, dbqa.as_retriever(), return_source_documents=True)
    question = "O que se trata esse site?"
    result = pdf_qa({"question": question})
    answer = result["answer"]
    
    st.write(answer)