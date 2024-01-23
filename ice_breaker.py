import os
import requests
import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from dotenv import load_dotenv
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile

load_dotenv()

if __name__ == "__main__":
    # debugging validator
    print("Hello Langchain!")

    # sets the Open API key from env
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
##########################################
    # MANUAL JSON DATA INPUT INSTEAD OF WASTING PROXYCURL API REQUESTS
    url = "https://gist.githubusercontent.com/binh-ngo/05fcc941df09ab8dcbb0d56b3f33e6ac/raw/9f1c723e902816084054e4bc9b83236250d96685/binh-ngo.json"

    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON content
        json_data = response.json()
    # END MANUAL DATA RETRIEVAL
##########################################
        
    # imports function from agent and sets user's linkedin profile url in variable
    linkedin_profile_url = linkedin_lookup_agent(name="Binh-Nguyen Ngo")

    # directions given to model
    summary_template = """
      given the Linkedin information {information} about a person, I want you to create:
      1. a short summary
      2. two interesting facts about them
    """

    # create prompt template with the directions and variables required
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # model settings
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # connects the directions and the model
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # imports function from third_parties and outputs linkedin profile info in json
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    linkedin_data = json_data
    # outputs the information requested from the prompt
    print(chain.run(information=linkedin_data))

# the run command allows you to run the chain as is
#  for invoke, you need to input a json object with input as the arg