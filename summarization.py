from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)


    st.image('img.jpg')
    st.subheader('Get a quick summary of your Documents!!')

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type=['txt'])
        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Uploading File'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                st.session_state.fl = file_name
        
    
    q = st.button('Get a brief summary of your document ')

    if q:
        if 'fl' in st.session_state:
            file_name = st.session_state.fl
            with open(f'{file_name}', encoding='utf-8') as f:
                text = f.read()
            llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
            chunks = text_splitter.create_documents([text])

            map_prompt = '''
                Write a short ant concise summary of the following:
                Text: `{text}`
                CONCISE SUMMARY:
                '''
            map_prompt_template = PromptTemplate(
                input_variables=['text'],
                template = map_prompt
            )

            combine_prompt = """
            write a concise summary of the following text that covers the key points.
            add a title to the summary.
            Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topics FOLLOWED by BULLET POINTS if possible and end the summary with a CONCLUSION PHRASE.
            Text: {text}
            """
            
            print(combine_prompt)
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

            summary_chain = load_summarize_chain(
                llm=llm,
                chain_type='map_reduce',
                map_prompt=map_prompt_template,
                combine_prompt=combine_prompt_template,
                verbose=False
            )
            output = summary_chain.run(chunks)
            st.text_area('LLM Answer: ', value=output)