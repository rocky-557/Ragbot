
from table import give_rel_tabs 
from img import get_imgs
from flo import gen_flo , genn
from model import gen_response
from vec_db import vecdb

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import matplotlib.pyplot as plt
import networkx as nx



# Set page configuration
st.set_page_config(page_title="Studz - An AI-based assistant for studies.")

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

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm Studz, How may I help you?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout containers
input_container = st.container()
response_container = st.container()

# User input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

with input_container:
    user_input = get_text()
    show_table = st.checkbox("Show table")
    show_flowchart = st.checkbox("Show flowchart")

# Generate response
def generate_response(query, show_table=False, show_flowchart=False):
    imgs = get_imgs(query=query)
    
    # Default responses
    response = gen_response(context=vecdb.similarity_search(question=query, k=6), query=query)
    
    table_df = None
    flowchart = None
    
    if show_flowchart:
        flowchart = genn(query)
    
    if show_table:
        table_df = give_rel_tabs(query) if flowchart is None else None
    
    return response, imgs, flowchart, table_df

# Display response
with response_container:
    if user_input:
        response, imgs, flowchart, table_df = generate_response(user_input, show_table, show_flowchart)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
        # Display chat messages
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
        
        # Display images
        if imgs:
            for i in imgs:
                st.image(i, caption="Relevant Image", use_column_width=True)
        
        # Display tables if requested
        if show_table and table_df is not None:
            st.write("Relevant Table:")
            st.dataframe(table_df)
        
        # Display flowchart if requested
        if show_flowchart and flowchart:
            G = nx.DiGraph()
            for node, data in flowchart['nodes'].items():
                G.add_node(node)
                for edge in data['edges']:
                    G.add_edge(node, edge)
            pos = nx.spring_layout(G)
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
            st.pyplot(plt)
