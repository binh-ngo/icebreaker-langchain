from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

information = """
  Elon Reeve Musk is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.
"""
if __name__ == "__main__":
  print("Hello Langchain!")

  openai_api_key = os.environ.get('OPENAI_API_KEY')
  summary_template = """
    given the information {information} about a person, I want you to create:
    1. a short summary
    2. two interesting facts about them
  """

  summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
  )

  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

  chain = LLMChain(llm=llm, prompt=summary_prompt_template)

  print(chain.invoke(input=information))