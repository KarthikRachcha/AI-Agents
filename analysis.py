import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType, agent
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)

np.random.seed(42)
n_rows=100

start_date = datetime(2024,1,1)
dates=[start_date + timedelta(days=i) for i in range(n_rows)]


makes = ['Toyota', 'Ford', 'Chevrolet', 'Honda', 'Nissan','BMW','Mercedes','Audi','Volkswagen','Hyundai','Kia']
models =['Sedan','SUV','Hatchback','Truck','Van','Convertible']
colors=['Red','Blue','Green','Yellow','Black','White','Gray','Silver','Brown','Orange']

data={
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

df= pd.DataFrame(data).sort_values('Date')

#print("\nFirst few rows of generated data:")
#print(df.head())

#print("\nDataframe Info:")
#print(df.info())

#print('\nSummary statistics:')
#print(df.describe())

agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,
agent_type=AgentType.OPENAI_FUNCTIONS,)

print("data analysis agent is ready. you can ask questions now about data.")


def ask_agent(question):
    """Function to ask questions to the agent and display the response"""
    response = agent.run({
        "input": question,
        "agent_scratchpad": f"Human: {question}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("---")


ask_agent("What are the column names in this dataset?")
ask_agent("How many rows are in this dataset?")
ask_agent("What is the average price of cars sold?")