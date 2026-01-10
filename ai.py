import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

# Page config
st.set_page_config(
    page_title="Research Paper Topic Recommender",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Research Paper Topic Recommender")
st.markdown("*AI-powered research topic suggestions tailored to your interests*")

# Sidebar for API key and settings
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.markdown("[Get your Groq API key](https://console.groq.com/keys)")
    
    st.divider()
    
    st.header("📊 Parameters")
    num_topics = st.slider("Number of topics to generate:", 3, 10, 5)
    research_level = st.selectbox(
        "Research Level:",
        ["Undergraduate", "Master's", "PhD", "Post-doctoral"]
    )
    
    st.divider()
    st.markdown("### About")
    st.info("This tool uses Groq AI to suggest relevant research paper topics based on your area of interest.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Your Research Interest")
    area_of_interest = st.text_area(
        "Describe your area of interest:",
        placeholder="e.g., Machine Learning in Healthcare, Sustainable Energy, Quantum Computing...",
        height=100
    )
    
    specific_focus = st.text_input(
        "Any specific focus? (Optional)",
        placeholder="e.g., neural networks, solar panels, error correction..."
    )
    
    current_trends = st.checkbox("Focus on current trends and emerging topics")
    interdisciplinary = st.checkbox("Include interdisciplinary topics")

# Function to generate topics
def generate_research_topics(api_key, interest, focus, level, num, trends, interdis):
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar!")
        return None
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        # Create prompt template
        template = """You are an expert research advisor with deep knowledge across multiple academic disciplines.

Research Interest: {interest}
Specific Focus: {focus}
Academic Level: {level}
Current Trends Focus: {trends}
Interdisciplinary Approach: {interdis}

Generate {num} compelling and feasible research paper topics. For each topic:
1. Provide a clear, concise title
2. Write a brief description (2-3 sentences) explaining the research gap and significance
3. Suggest potential research methodologies
4. Indicate expected impact and relevance

Format your response as follows for each topic:

**Topic 1: [Title]**
**Description:** [Description]
**Methodology:** [Suggested methods]
**Impact:** [Expected impact and relevance]

---

Make sure the topics are:
- Original and innovative
- Feasible within the academic level specified
- Relevant to current research needs
- Clear and well-defined
- Number each topic sequentially (Topic 1, Topic 2, etc.)"""

        prompt = PromptTemplate(
            input_variables=["interest", "focus", "level", "num", "trends", "interdis"],
            template=template
        )
        
        # Format the prompt
        formatted_prompt = prompt.format(
            interest=interest,
            focus=focus if focus else "No specific focus",
            level=level,
            num=num,
            trends="Yes, focus on current trends" if trends else "No specific trend focus",
            interdis="Yes, include interdisciplinary approaches" if interdis else "Focus on single discipline"
        )
        
        # Generate response directly
        response = llm.invoke(formatted_prompt)
        
        return response.content
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Generate button
with col1:
    generate_btn = st.button("🚀 Generate Research Topics", type="primary", use_container_width=True)

# Results section
with col2:
    st.subheader("💡 Suggested Research Topics")
    
    if generate_btn:
        if not area_of_interest:
            st.warning("Please describe your area of interest first!")
        else:
            with st.spinner("Generating research topics... This may take a moment."):
                topics = generate_research_topics(
                    groq_api_key,
                    area_of_interest,
                    specific_focus,
                    research_level,
                    num_topics,
                    current_trends,
                    interdisciplinary
                )
                
                if topics:
                    st.markdown(topics)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Topics",
                        data=topics,
                        file_name="research_topics.txt",
                        mime="text/plain"
                    )
    else:
        st.info("👈 Enter your research interest and click 'Generate Research Topics' to get started!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit, LangChain, and Groq AI | 
    Remember to validate topics with your advisor and check existing literature</p>
</div>
""", unsafe_allow_html=True)