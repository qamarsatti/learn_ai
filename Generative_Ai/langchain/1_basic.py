from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
openkey  = ""

llm = OpenAI(openai_api_key=openkey)
# chat_model = ChatOpenAI()


text = "hi"

l=llm.predict(text)
print(l)

