import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
load_dotenv()
llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)


template = """ you are a helpful assitant. your task is to answer the user's question to the best ofyour ability.

User's question: {question}

please provide a lcear and concise answer:
"""

prompt= PromptTemplate(template=template, input_variables=["question"])


qa_chain = prompt | llm

def get_answer(question):
    """
    get an answer to the given question using the QA chian.
    """
    input_variables = {"question": question}
    response = qa_chain.invoke(input_variables).content
    return response

user_question = input("Enter your question: ")
user_answer = get_answer(user_question)
print(f"Answer: {user_answer}")