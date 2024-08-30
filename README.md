# RAG Implementation with LLaMA LLM , CLIP

This project provides an implementation of a Retrieval-Augmented Generation (RAG) model using the LLaMA language model (LLM) , FAISS vector db and Groq. The RAG architecture combines dense retrieval with generative language modeling to create a powerful question-answering system. In this implementation, Groq accelerates the LLaMA LLM, enabling high-performance inference.

### !!! Note : The given PDFS are not provided in here .  Kindly Download / add to the project folder and change the name respectively in the vec_db.py file. Also all the pdfs are not utilized till now . Only Vol-3 is used. With minor changes We can include all the PDFs . This is a Preventive Measure Taken to avoid too much Ahead Of Time (AOT) delay at the time of development. 
### Kindly USE app2.py for testing . The Given Groq Api key is Still working , and will not be revoked until next Announcement.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)


## Introduction

Retrieval-Augmented Generation (RAG) models use an external retriever to fetch relevant documents from a knowledge base, and a generative language model (like LLaMA) to generate responses based on both the retrieved documents and the input query. This implementation leverages the Groq platform for efficient inference, enhancing the speed and scalability of the LLaMA LLM.

## Features

- **FAISS Retrieval**: Uses a dense retriever for retrieving relevant documents from a knowledge base using FAISS .
- **Groq Acceleration**: Accelerates inference using Groq for high-performance processing.
- **Easy Configuration**: Simple setup and configuration for different datasets and use cases.
- **Scalable**: Supports scaling across multiple Groq LPU chips for larger workloads.
- **CLIP**: Supports to retrieve text from images.

## Prerequisites

Before using this implementation, ensure you have the following:

- **Python 3**: The code is written in Python, so a compatible version is required.
- **Groq SDK**: You need the Groq SDK installed on your machine. Visit the [Groq website](https://groq.com/) for more information.
- **Dependencies**: Install required dependencies listed in `requirements.txt`.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/rocky-557/Studz-bot.git
    cd Studz-bot
    ```

2. **Install Dependencies:**
3. *The full list is not updated yet , because of the COLAB environment !!*
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Groq SDK:**
    Follow the instructions from the [Groq SDK documentation](https://groq.com/docs/sdk) to set up the environment.



