
import streamlit as st
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Advanced Agentic RAG",
    page_icon="🧠",
    layout="wide"
)

URLS = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing"
]

# Globals (populated in main)
retriever = None
llm = None


class ChatMemory:
    """Simple chat memory system"""
    def __init__(self):
        self.messages: list[dict] = []
        self.context_window = 10  # Keep last 10 messages

    def add_message(self, role: str, content: str):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Trim history
        if len(self.messages) > self.context_window:
            self.messages = self.messages[-self.context_window :]

    def get_context(self) -> str:
        if not self.messages:
            return "No previous conversation."

        context = "Previous conversation context:\n"
        for msg in self.messages[-6:]:  # Last 3 exchanges = 6 msgs
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context

    def clear(self):
        self.messages = []


class TaskDecomposer:
    """Breaks down complex queries into sub‑tasks"""
    def __init__(self, llm):
        self.llm = llm

    def decompose_task(self, query: str) -> List[str]:
        prompt = f"""
Analyze this query and break it down into smaller, manageable sub‑tasks if it's complex.
If it's a simple query, return it as is.

Query: {query}

Return a JSON list of sub‑tasks. For simple queries, return a single‑item list.
Example: ["task1", "task2", "task3"]

Sub‑tasks:"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # In production prefer json.loads
            return json.loads(content) if content.startswith("[") else [query]
        except Exception:
            return [query]


class SelfReflection:
    """Self‑reflection system for the agent"""
    def __init__(self, llm):
        self.llm = llm
        self.performance_history: List[Dict[str, Any]] = []

    def reflect_on_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        reflection_prompt = f"""
Reflect on this AI response and evaluate its quality:

Original Query: {query}
AI Response: {response}
Context Used: {context[:500]}...

Evaluate on a scale of 1‑10:
1. Relevance to the query
2. Use of provided context
3. Completeness of answer
4. Clarity and coherence

Also suggest improvements if needed.

Return JSON format:
{{
  "relevance_score": 0‑10,
  "context_usage_score": 0‑10,
  "completeness_score": 0‑10,
  "clarity_score": 0‑10,
  "overall_score": 0‑10,
  "improvements": "suggestions",
  "confidence": "high/medium/low"
}}
"""
        try:
            reflection = self.llm.invoke(reflection_prompt)
            self.performance_history.append(
                {"reflection": reflection.content, "timestamp": datetime.now().isoformat()}
            )
            return self.performance_history[-1]
        except Exception:
            return {"reflection": "Unable to reflect on response", "timestamp": datetime.now().isoformat()}

    def get_performance_summary(self) -> str:
        if not self.performance_history:
            return "No performance history available."
        recent = self.performance_history[-5:]
        return f"Recent performance: {len(recent)} evaluations."



@st.cache_resource
def initialize_llm():
    """Create LLM (cached)"""
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1000,
    )


@st.cache_resource
def create_knowledge_base(urls):
    """Create Chroma vector store from list of URLs"""
    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def _get_retriever():
    return st.session_state.get("retriever", retriever)


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents from the knowledge base."""
    try:
        docs = _get_retriever().get_relevant_documents(query)
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs[:4], 1):
            content = doc.page_content[:600]
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"Document {i} (Source: {source}):\n{content}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


@tool
def search_specific_topic(topic: str) -> str:
    """Search for specific information about a topic in the knowledge base."""
    try:
        expanded_query = f"{topic} definition explanation examples applications"
        docs = _get_retriever().get_relevant_documents(expanded_query)
        if not docs:
            return f"No specific information found about {topic}."
        return f"Specific information about {topic}:\n{docs[0].page_content[:800]}"
    except Exception as e:
        return f"Error searching for {topic}: {str(e)}"


@tool
def analyze_context_relevance(query: str, context: str) -> str:
    """Analyze how well the retrieved context matches the query."""
    try:
        llm_local = st.session_state.llm
        prompt = f"""
Analyze how relevant this context is to answering the query:

Query: {query}
Context: {context[:500]}...

Rate relevance (1‑10) and explain why. Also suggest if we need additional information.

Analysis:"""
        response = llm_local.invoke(prompt)
        return f"Context Analysis: {response.content}"
    except Exception as e:
        return f"Error analyzing context: {str(e)}"


@tool
def memory_recall(topic: str) -> str:
    """Recall previous conversations about a specific topic."""
    try:
        if "chat_memory" not in st.session_state:
            return "Chat memory not initialized."

        memory_context = st.session_state.chat_memory.get_context()
        llm_local = st.session_state.llm

        recall_prompt = f"""
Search through this conversation history for information related to: {topic}

Conversation History:
{memory_context}

Extract and summarize any relevant previous discussions about {topic}.
If no relevant information found, return "No previous discussion about this topic."

Relevant Information:"""
        response = llm_local.invoke(recall_prompt)
        return f"Memory Recall: {response.content}"
    except Exception as e:
        return f"Error recalling memory: {str(e)}"


@tool
def autonomous_decision_maker(query: str, available_info: str) -> str:
    """Make autonomous decisions about how to proceed with the query."""
    try:
        llm_local = st.session_state.llm
        prompt = f"""
You are an autonomous decision‑making system. Analyze this situation and decide the best approach:

User Query: {query}
Available Information: {available_info[:300]}...

Decide whether to:
1. Answer directly with available information
2. Search for more specific information
3. Break down the query into sub‑parts
4. Ask for clarification
5. Combine multiple approaches

Provide your decision and reasoning.

Decision:"""
        response = llm_local.invoke(prompt)
        return f"Autonomous Decision: {response.content}"
    except Exception as e:
        return f"Error in decision making: {str(e)}"



def create_advanced_agent():
    tools = [
        retrieve_documents,
        search_specific_topic,
        analyze_context_relevance,
        memory_recall,
        autonomous_decision_maker,
    ]

    agent_prompt = PromptTemplate.from_template(
        """
You are an advanced AI assistant with autonomous decision‑making, memory, and self‑reflection capabilities.

Your capabilities:
- Remember previous conversations
- Make autonomous decisions about information gathering
- Decompose complex tasks
- Reflect on your responses
- Dynamically gather information

Available tools:
{tools}

Current Query: {input}
Previous Actions: {agent_scratchpad}

Memory Context: {memory_context}

Instructions:
1. Consider the conversation history when relevant
2. Make autonomous decisions about which tools to use
3. If the query is complex, break it down into sub‑tasks
4. Gather information dynamically based on what you discover
5. Be thorough but efficient

Respond with valid JSON in one of these formats:

To use a tool:
{{"action": "tool_name", "action_input": "your_input_here"}}

For final answer:
{{"action": "Final Answer", "action_input": "your_complete_answer_here"}}

Your JSON response:"""
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "tools": lambda x: render_text_description_and_args(tools),
            "memory_context": lambda x: st.session_state.chat_memory.get_context()
            if "chat_memory" in st.session_state
            else "No memory",
        }
        | agent_prompt
        | st.session_state.llm  # Always pull from session_state
        | JSONAgentOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def main():
    global retriever, llm

    st.title("🧠 Advanced Agentic RAG System")
    st.write(
        "An AI assistant with memory, autonomous decision‑making, and self‑reflection capabilities!"
    )

    # Session defaults
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ChatMemory()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your advanced AI assistant with memory and autonomous capabilities."
                    " What would you like to explore?"
                ),
            }
        ]

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Sidebar 
    with st.sidebar:
        st.header("🎛️ System Controls")

        msg_count = len(st.session_state.chat_memory.messages)
        st.success(f"💭 Memory: {msg_count} messages" if msg_count else "💭 Memory: Empty")

        if st.button("🗑️ Clear Memory"):
            st.session_state.chat_memory.clear()
            st.success("Memory cleared!")

        if "reflection_system" in st.session_state:
            st.info(f"📊 {st.session_state.reflection_system.get_performance_summary()}")


    # Initialize heavy components
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing advanced system..."):
            try:
                llm = initialize_llm()
                st.session_state.llm = llm

                if "reflection_system" not in st.session_state:
                    st.session_state.reflection_system = SelfReflection(llm)

                if "task_decomposer" not in st.session_state:
                    st.session_state.task_decomposer = TaskDecomposer(llm)

                vectorstore = create_knowledge_base(URLS)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.session_state.retriever = retriever

                st.session_state.agent = create_advanced_agent()
                st.session_state.initialized = True
                st.success("🎉 Advanced system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
                return

    # Chat UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_memory.add_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking autonomously..."):
                try:
                    sub_tasks = st.session_state.task_decomposer.decompose_task(prompt)
                    if len(sub_tasks) > 1:
                        st.info(f"🔄 Breaking down into {len(sub_tasks)} sub‑tasks...")
                        with st.expander("📋 Sub‑tasks identified"):
                            for i, task in enumerate(sub_tasks, 1):
                                st.write(f"{i}. {task}")

                    result = st.session_state.agent.invoke({"input": prompt})
                    response = result["output"]
                    st.write(response)

                    if result.get("intermediate_steps"):
                        with st.expander("🔍 Agent's reasoning process"):
                            for i, (action, observation) in enumerate(
                                result["intermediate_steps"], 1
                            ):
                                st.write(f"**Step {i}:** Used `{action.tool}`")
                                st.write(f"*Input:* {action.tool_input}")
                                st.write(f"*Result:* {observation[:300]}...")
                                st.divider()

                    context_used = str(result.get("intermediate_steps", ""))[:500]
                    reflection = st.session_state.reflection_system.reflect_on_response(
                        prompt, response, context_used
                    )
                    with st.expander("🪞 Self‑reflection"):
                        st.write(reflection["reflection"])

                    st.session_state.chat_memory.add_message("assistant", response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"🚨 I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
