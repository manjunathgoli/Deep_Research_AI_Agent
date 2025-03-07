# Deep Research AI Agentic System  

## 📌 Overview  
This project is an AI-powered research system that fetches online data, expands on unknown terms, summarizes content, and generates answers using a multi-agent approach. It is built using **LangChain**, **LangGraph**, **Tavily Search API**, and **Facebook's BART-Large model** for summarization.  

## 🚀 How It Works  
The system follows these steps to answer a research query:  
1. **Research Agent** → Fetches relevant web search results.  
2. **Term Expansion Agent** → Identifies and defines unknown terms (if needed).  
3. **Summarization Agent** → Processes the research data and creates a concise summary.  
4. **Answer Agent** → Forms a structured answer and provides source links.  

---

## 🛠️ Installation  

Make sure you have **Python 3.8+** installed. Then, run the following:  

```bash
pip install langchain-community langgraph torch transformers
```
You also need to set up a Tavily API Key:

```bash
export TAVILY_API_KEY="your_api_key_here"
```

📂 Project Structure
```bash
📂 deep-research-ai
 ├── research_system.py  # Main script
 ├── README.md           # Documentation
 ├── requirements.txt    # Dependencies
 ├── .env                # API Keys (Optional)
```

🏗️ Code Explanation
1️⃣ Imports & Setup
```bash
import os
import torch
from langchain.tools import TavilySearchResults
from langgraph.graph import StateGraph
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```

1. os → Manages environment variables
2. torch → Checks if GPU is available for faster model execution
3. TavilySearchResults → Handles online research queries
4. StateGraph → Defines the AI workflow
5. transformers → Loads the BART-Large model for summarization


2️⃣ Initialize Model & Device
```bash
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
Loads BART for summarization

Uses GPU if available for faster processing

3️⃣ Research Agent (Fetches online data)
```bash
class ResearchAgent:
    def __call__(self, state):
        search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
        query = state["question"]
        results = search_tool.run(query)
        
        state["research_data"] = results or []
        state["source_urls"] = [r["url"] for r in results if "url" in r]  
        return state
```
Uses Tavily API to fetch search results

Stores research data and source URLs

4️⃣ Term Expansion Agent (Finds unknown terms)
```bash
class TermExpansionAgent:
    def __call__(self, state):
        unknown_terms = extract_unknown_terms(state["research_data"], state["question"])
        if unknown_terms:
            search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
            state["definitions"] = {term: search_tool.run(term) for term in unknown_terms}
        return state
```
Finds complex terms in research data

Fetches definitions using Tavily

5️⃣ Summarization Agent (Creates a short summary)
```bash
class SummarizationAgent:
    def __call__(self, state):
        context = " ".join(str(item) for item in state.get("research_data", []))
        input_text = "summarize: " + context
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        if inputs.input_ids.shape[1] == 0:
            state["summary"] = "No relevant research data available."
            return state

        summary_ids = model.generate(**inputs, max_length=512, min_length=150)
        state["summary"] = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return state
```
Converts research data into a short summary

Uses BART model for text summarization

6️⃣ Answer Agent (Generates final answer)
```bash
class AnswerAgent:
    def __call__(self, state):
        context = " ".join(str(item) for item in state.get("research_data", []))
        input_text = "summarize: " + context
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

        if inputs.input_ids.shape[1] == 0:
            state["answer_text"] = "No content available for answering."
            return state

        summary_ids = model.generate(**inputs, max_length=512, min_length=50)
        state["answer_text"] = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        if state.get("source_urls"):
            state["answer_text"] += "\n\nSources:\n" + "\n".join(state["source_urls"])
        
        return state
```
Formats final answer using summarization

Adds source links for credibility

7️⃣ Building the AI Workflow
```bash
from langgraph.graph import StateGraph
graph = StateGraph(State)
graph.add_node("research", ResearchAgent())
graph.add_node("term_expansion", TermExpansionAgent())
graph.add_node("summarization", SummarizationAgent())
graph.add_node("answer", AnswerAgent())
```
Creates a state machine for multi-agent processing

🌐 Defining Transitions
```bash
graph.add_conditional_edges(
    "research",
    lambda state: "term_expansion" if extract_unknown_terms(state["research_data"], state["question"]) else "summarization",
)
graph.add_edge("term_expansion", "summarization")
graph.add_edge("summarization", "answer")
graph.set_entry_point("research")
compiled_graph = graph.compile()
```
Automatically expands terms if needed

Ensures proper order of execution

8️⃣ Running the System
```bash
def runsystem(query):
    state = {
        "question": query,
        "research_data": [],
        "definitions": {},
        "summary": "",
        "answer_text": "",
        "source_urls": [],
    }
    final_state = compiled_graph.invoke(state)

    print("\nGenerated Answer:\n")
    print(final_state.get("answer_text", "No answer generated."))
```
Takes user input

Runs the AI system

Prints the final answer

📊 Example Run
```bash
$ python research_system.py
Enter your research question: What are the latest trends in AI?
```

Output:
```bash
Generated Answer:
The latest trends in AI include advancements in AGI, LLMs, and real-time processing.

Sources:
https://example.com/ai-trends-2025
https://example.com/deep-learning
```

🎯 Future Improvements
1. Add vector embeddings for better contextual understanding
2. Improve term extraction for more accurate expansion
3. Introduce interactive UI using Streamlit or Flask

🏆 Credits
1. LangChain & LangGraph for AI Workflow
2. Facebook's BART Model for summarization
3. Tavily API for real-time research



