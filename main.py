# main_6 : pdf파일 읽기 + splitting + embedding + db 저장 + 챗봇 기능 추가+ web 서비스 + db 업데이트
from langchain_chroma import Chroma
import shutil
import tempfile
import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import hashlib
from dotenv import load_dotenv
load_dotenv()


# Chroma DB가 저장된 디렉토리 경로
persist_directory = './db/chromadb'
hash_file_path = './db/pdf_hash.txt'

# Streamlit 웹페이지 제목 설정
st.title("ChatPDF")
st.write("---")

# 파일 업로드 기능 구현
uploaded_file = st.file_uploader("업로드할 파일을 선택해 주세요.")
st.write("---")

# PDF 파일을 처리하는 함수 (PDF를 임시 폴더에 저장 후 페이지별로 로드)


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 파일 해시를 계산하는 함수


def calculate_file_hash(file):
    hasher = hashlib.sha256()
    hasher.update(file.getvalue())
    return hasher.hexdigest()

# 해시 파일에 저장된 이전 해시를 불러오는 함수


def load_previous_hash():
    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            return f.read()
    return None

# 새로운 해시값을 해시 파일에 저장하는 함수


def save_current_hash(hash_value):
    with open(hash_file_path, 'w') as f:
        f.write(hash_value)

# Chroma DB를 초기화하고 기존 데이터를 덮어쓰지 않고 사용 중인 DB에 새 데이터 추가


def initialize_chroma_db(texts, embeddings_model):
    if os.path.exists(persist_directory):
        # 기존 Chroma DB 로드
        chromadb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
            collection_name='esg'
        )
    else:
        # Chroma DB 새로 생성
        chromadb = Chroma.from_documents(
            texts,
            embeddings_model,
            collection_name='esg',
            persist_directory=persist_directory,
        )
    return chromadb


# PDF 파일이 업로드되면 실행되는 코드
if uploaded_file is not None:
    try:
        # st.write("Processing uploaded PDF file...")

        # 새로 업로드된 파일의 해시값 계산
        current_file_hash = calculate_file_hash(uploaded_file)
        previous_file_hash = load_previous_hash()

        embeddings_model = OpenAIEmbeddings()

        # 기존 파일과 새로운 파일의 해시가 다르면 DB를 새로 생성
        if current_file_hash != previous_file_hash:
            st.info("PDF 파일이 업로드되었습니다. DB를 새로 생성합니다.")

            # PDF 파일을 로드하여 페이지별로 분리
            pages = pdf_to_document(uploaded_file)
            # st.write("PDF loaded and split successfully.")

            # 텍스트 분리 설정 (1000자씩 분리하고, 50자의 오버랩 적용)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_documents(pages)
            # st.write("Text splitting completed.")

            # Chroma 데이터베이스 초기화 또는 기존 DB와 병합
            chromadb = initialize_chroma_db(texts, embeddings_model)
            # st.write("Chroma DB initialized successfully.")

            # 새로운 파일의 해시값을 저장
            save_current_hash(current_file_hash)
        else:
            # st.write("Loading existing Chroma DB...")
            chromadb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings_model,
                collection_name='esg'
            )

        # 질문을 받을 수 있는 섹션 생성
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')

        # 질문하기 버튼 클릭 시 동작
        if st.button('질문하기'):
            with st.spinner('Wait for it...'):
                llm = ChatOpenAI(model_name="gpt-4o", temperature=2)
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=chromadb.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                result = qa_chain.invoke({"query": question})
                st.write(result["result"])

    except Exception as e:
        st.error(f"An error occurred: {e}")
