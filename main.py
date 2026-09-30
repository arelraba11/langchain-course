from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
    """

    summary_template = """  
        given the information about a person i want you create:
        1. A short summary of the person in 2-3 sentences.
        2. A 2 key skills or strengths of the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm_cloud = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm_local = ChatOllama(model="granite4.1:8b", temperature=0)

    chain = summary_prompt_template | llm_cloud
    response = chain.invoke(input={"information": information})
    print(response.content )


if __name__ == "__main__":
    main()
