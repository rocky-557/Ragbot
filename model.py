#set GROQ_API_KEY in the secrets

import os
from groq import Groq

# Create the Groq client
client = Groq(api_key='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', )

'''# Kindly note this that the api key available wont work . 
Please Sign Up to Groq and replace the API key or put it in env variable "GROQ_API_KEY"

'''
# Set the system prompt
system_prompt = {
    "role": "system",
    "content": 
    "You are an study assistant for students .You reply shortly , consisely ,no extra."
}

# Initialize the chat history
chat_history = [system_prompt]

model='llama-3.1-8b-instant'


def gen_response(context='unavailable',query='introduce urself'):
  '''
  A Function for generating response from the required model using the context from RAG.
  '''
  user_input = f'context:{context} , Answer the query:{query}'
  # Append the user input to the chat history
  chat_history.append({"role": "user", "content": user_input})

  response = client.chat.completions.create(model=model,
                                            messages=chat_history,
                                            max_tokens=300,
                                            temperature=1.05)
  # Append the response to the chat history
  chat_history.append({
      "role": "assistant",
      "content": response.choices[0].message.content
  })
  # return the response
  return response.choices[0].message.content

print(gen_response())

