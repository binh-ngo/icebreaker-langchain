from tools.tools import get_profile_url
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool, AgentType


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given the full name {name_of_person} I want you to give me a link to their Linkedin profile page.
                      Your answer should only contain a URL"""

    prompt = hub.pull("hwchase17/react")

    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the Linkedin URL",
        )
    ]
    #  Zero shot react description agent is the default and verbose describes every step of the process
    agent = create_react_agent(tools=tools_for_agent, llm=llm, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        return_intermediate_steps=True,
        verbose=True,
        handle_parsing_errors=True,
    )

    # agent_executor.invoke({"input": "hi"})

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linkedin_profile_url = agent_executor.invoke(
        {
            "input": f"Given this name: {name}, I want you to find the LinkedIn of the person and your answer should contain only the URL of the LinkedIn profile"
        }
    )

    return linkedin_profile_url
