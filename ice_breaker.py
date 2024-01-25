import os
import requests
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from dotenv import load_dotenv
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from output_parsers import person_intel_parser, PersonIntel
from typing import Tuple

load_dotenv()

def ice_break(name: str) -> Tuple[PersonIntel, str]:
    ##########################################
    # MANUAL JSON DATA INPUT INSTEAD OF WASTING PROXYCURL API REQUESTS
    binh_url = "https://gist.githubusercontent.com/binh-ngo/05fcc941df09ab8dcbb0d56b3f33e6ac/raw/9f1c723e902816084054e4bc9b83236250d96685/binh-ngo.json"
    patrick_url = 'https://gist.githubusercontent.com/binh-ngo/feeccc4331c7ef9634584301e9c85719/raw/1f1b6e20ba7adfb17b54ae8aa020b900ee6ca9a1/patrick.json'

    # Make a GET request to the URL
    # response = requests.get(url)

    # Check if the request was successful (status code 200)
    # if response.status_code == 200:
    # Parse the JSON content
    # json_data = response.json()

    # linkedin_data = json_data

    # END MANUAL DATA RETRIEVAL
    ##########################################

    # sets the Open API key from env
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # imports function from agent and sets user's linkedin profile url in variable
    linkedin_username = linkedin_lookup_agent(name=name)
    # print("linkedinURL:", linkedin_username)
    # linkedin_username returned the whole json object that looks like this
    # linkedinURL: {
    #     "input": "Given this name: Binh-Nguyen Ngo, I want you to find the LinkedIn account of the person and your final answer should contain only the URL of the LinkedIn profile. MAke sure the format of the url is like this https://www.linkedin.com/in/username",
    #     "output": "The LinkedIn profile of Binh-Nguyen Ngo is https://www.linkedin.com/in/binh-nguyen-ngo/",
    #     "intermediate_steps": [
    #         (
    #             AgentAction(
    #                 tool="Crawl Google for linkedin profile page",
    #                 tool_input="Binh-Nguyen Ngo LinkedIn",
    #                 log='I should use the tool "Crawl Google for linkedin profile page" to find the LinkedIn profile of Binh-Nguyen Ngo.\nAction: Crawl Google for linkedin profile page\nAction Input: Binh-Nguyen Ngo LinkedIn',
    #             ),
    #             "https://www.linkedin.com/in/binh-nguyen-ngo/",
    #         )
    #     ],
    # }

    # imports function from third_parties and outputs linkedin profile info in json
    linkedin_data = scrape_linkedin_profile(
        # needed to extract the url from above
        linkedin_profile_url=linkedin_username["output"].split(" ")[-1]
    )

    # url = binh_url
    # response = requests.get(url)
    # if response.status_code == 200:
    #     json_data = response.json()
    # linkedin_data = json_data
    # print('ICEBREAKER linkedin_data', linkedin_data)

    # directions given to model
    summary_template = """
      given the Linkedin information {information} about a person, I want you to create:
      1. a short summary
      2. two interesting facts about them
      3. A topic that may interest them
      4. 2 creative icebreakers to open a conversation with them
      \n{format_instructions}
    """

    # create prompt template with the directions and variables required
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    # model settings
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # connects the directions and the model
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # outputs the information requested from the prompt
    result = chain.run(information=linkedin_data)
    print(person_intel_parser.parse(result), linkedin_data.get("profile_pic_url"))
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")


# the run command allows you to run the chain as is
#  for invoke, you need to input a json object with input as the arg

if __name__ == "__main__":
    # debugging validator
    # result = ice_break(name="Patrick Chour")
    # print('ICEBREAKER:',result)
    pass
