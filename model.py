#set GROQ_API_KEY in the secrets

import os
from groq import Groq

# Create the Groq client
client = Groq(api_key='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', )



# Set the system prompt
system_prompt = {
    "role": "system",
    "content": 
    "You are an study assistant for students .You reply consisely , briefly .if needed explain . no extra."
}

# Initialize the chat history
chat_history = [system_prompt]

model='gemma-7b-it'


def gen_response(context='unavailable',query='introduce urself',maxx=300):
  '''
  A Function for generating response from the required model using the context from RAG.
  '''
  user_input = f'context:{context} , Answer the query:{query}'
  # Append the user input to the chat history
  chat_history.append({"role": "user", "content": user_input})

  response = client.chat.completions.create(model=model,
                                            messages=chat_history,
                                            max_tokens=maxx,
                                            temperature=1.05)
  # Append the response to the chat history
  chat_history.append({
      "role": "assistant",
      "content": response.choices[0].message.content
  })
  # return the response
  return response.choices[0].message.content



