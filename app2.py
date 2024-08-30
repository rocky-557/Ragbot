from vec_db import vecdb
from model import gen_response
import streamlit as st
from img import get_imgs
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(page_title="Studz - An Ai based assistant for studies.")

# Sidebar contents
with st.sidebar:
    st.title('💬 Studz ')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - LLM model - Llama 💡
    ''')
    add_vertical_space(5)
    st.write('-- Made by Sri Raghav and Ranganathan - ECE G1 2nd Yr -- ')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm Studz, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(query):
    imgs=get_imgs(query=query)
    response = gen_response(context=vecdb.similarity_search(question=query,k=6),query=query)
    return response,imgs

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response,imgs = generate_response(user_input)
        st.session_state.past.append(user_input)
        for i in imgs:
            st.image(i, caption=f"Relevant Images That I Found", use_column_width=True)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))