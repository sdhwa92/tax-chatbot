from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_ai_message(user_question):

    # 데이터베이스 초기화
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "kor-tax-with-markdown"
    database = PineconeVectorStore.from_existing_index(index_name, embedding)

    # 모델 초기화
    llm = ChatOpenAI(model="gpt-4o")

    # 프롬프트 가져오기
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")
    # 검색 쿼리 처리
    database.as_retriever(search_kwargs={"k": 2})

    # 질문 처리
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=database.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )

    # 사전 처리
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # Define the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 한국의 소득세 전문가 입니다, 유저의 질문을 보고 답해주세요.",
            ),
            (
                "user",
                f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요.
            
            사전: {dictionary}
            
            질문: {{question}}
        """,
            ),
        ]
    )

    # Define the Output Parser
    output_parser = StrOutputParser()

    # Create the Chain
    dictionary_chain = prompt | llm | output_parser

    # 체인 생성
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_question})
    return ai_message["result"]
