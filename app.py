import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = '请输入你的OPENAI-KEY'

def main():

    # UI界面的标题
    st.set_page_config(page_title="Chat with your PDF", initial_sidebar_state="auto", layout="wide")
    st.header("开始和你的PDF聊天吧(*╹▽╹*)")

    # 配置temperature参数
    st.sidebar.subheader("配置参数")
    temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, step=0.1)


    # upload file
    pdf = st.file_uploader("Uploader your PDF document!!!", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",  # 用于分割文本的分隔符。这里使用\n作为分隔符，表示将文本按照换行符分割成多个块。
            chunk_size=1500,  # 每个块的大小，以字符数为单位。这里将块大小设置为1000，表示将文本分割成1000个字符的块。
            chunk_overlap=200,  # 相邻块之间的重叠量，以字符数为单位。这里将重叠量设置为200，表示相邻块之间有200个字符是重叠的。
            length_function=len  # 用于计算文本长度的函数。这里使用len函数，表示将文本长度定义为字符数。
        )
        texts = text_splitter.split_text(raw_text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)


        # create template
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know or you can't find the answer in the article, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in Chinese:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"])

        # show user input
        user_question = st.text_input("Now,chatting with your PDF!-!:")
        if user_question:
            with get_openai_callback() as cb:
                docs = docsearch.similarity_search(user_question)
                llm = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo-0301")
                chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
                res = chain.run(input_documents=docs, question=user_question)
                st.write(res)
                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Successful Requests: {cb.successful_requests}")
                st.write(f"Total Cost (USD): ${cb.total_cost}")

if __name__ == '__main__':
    main()
