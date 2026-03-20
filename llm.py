from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "kor-tax-with-markdown"
    database = PineconeVectorStore.from_existing_index(index_name, embedding)
    retriever = database.as_retriever(search_kwargs={"k": 2})
    return retriever


def get_llm(model="gpt-4o"):
    return ChatOpenAI(model=model)


def get_dictionary_chain():
    # 사전 처리
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

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
            
            질문: {{input}}
        """,
            ),
        ]
    )

    # Define the Output Parser
    output_parser = StrOutputParser()

    # Create the Chain
    dictionary_chain = prompt | llm | output_parser

    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def get_ai_response(user_question, session_id):
    # 사전 처리 체인 생성
    dictionary_chain = get_dictionary_chain()

    # 질문 처리
    rag_chain = get_rag_chain()

    # 체인 생성
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {"input": user_question}, config={"configurable": {"session_id": session_id}}
    )
    return ai_response
