import streamlit as st
import pandas as pd
import faiss
import os
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="AI Tool Finder", page_icon="🔍")

# Load tools
@st.cache_data
def load_tools():
    return pd.read_csv("tools.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_gpt_explanation(query, tool_info):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI tool expert. Explain why a specific AI tool would be good for a user's needs in 2-3 concise sentences."},
                {"role": "user", "content": f"Query: '{query}'\nTool: {tool_info['name']}\nDescription: {tool_info['description']}\nCategory: {tool_info['category']}\n\nWhy is this tool a good match?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Unable to generate explanation at the moment."

# Load data and model
df = load_tools()
model = load_model()

# Create FAISS index
@st.cache_resource
def create_faiss_index():
    tool_descriptions = df['description'].tolist()
    embeddings = model.encode(tool_descriptions)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

index = create_faiss_index()

# UI
st.title("🔍 AI Tool Finder")
st.markdown("""
Describe what you want to do, and I'll find the most relevant AI tools for you!
""")

# Input query
query = st.text_input(
    "What do you want an AI tool to do?",
    placeholder="e.g., Convert voice recordings to text"
)

if query:
    with st.spinner("🔍 Searching for the best AI tools..."):
        # Get embeddings and search
        query_vec = model.encode([query])
        _, indices = index.search(query_vec, k=3)
        
        st.subheader("🎯 Top Matching Tools")
        
        for idx in indices[0]:
            tool = df.iloc[idx]
            
            # Create a card-like container for each tool
            with st.container():
                st.markdown(f"### [{tool['name']}]({tool['link']})")
                st.markdown(f"**Category**: {tool['category']}")
                st.markdown(f"**Description**: {tool['description']}")
                
                # Get and display GPT explanation
                with st.spinner("Getting AI explanation..."):
                    explanation = get_gpt_explanation(query, tool)
                    st.markdown("**Why this matches your needs:**")
                    st.markdown(f"*{explanation}*")
                
                st.markdown("---") 