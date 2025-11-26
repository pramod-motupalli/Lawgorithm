import streamlit as st
import os
from lawgorithm.agents import RouterAgent, CaseParsingAgent, RetrievalAgent, VerdictAgent
from lawgorithm import config

# Page Config
st.set_page_config(
    page_title="Lawgorithm | AI Verdict Predictor",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .verdict-card {
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 15px;
        color: white;
    }
    
    .outcome-plaintiff { background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); }
    .outcome-defendant { background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%); }
    .outcome-neutral { background: linear-gradient(90deg, #8e9eab 0%, #eef2f3 100%); color: #333; }
    
    h1, h2, h3 { color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("⚖️ Lawgorithm")
st.subheader("Agentic AI System for Probable Verdict Generation")
st.markdown("---")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", value=config.GROQ_API_KEY or "", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        config.GROQ_API_KEY = api_key # Update config as well
    
    st.info("This system uses a Multi-Agent Architecture:\n\n1. **Router Agent**: Classifies case type.\n2. **Parsing Agent**: Structures facts.\n3. **Retrieval Agent**: Finds precedents (RAG).\n4. **Verdict Agent**: Predicts outcome.")

from lawgorithm.analytics import AnalyticsEngine

# Initialize Agents (Cached)
@st.cache_resource
def load_retrieval_agent():
    return RetrievalAgent()

@st.cache_resource
def load_reasoning_agents():
    return RouterAgent(), CaseParsingAgent(), VerdictAgent()

@st.cache_resource
def load_analytics_engine():
    return AnalyticsEngine()

# Main Form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Case Details")
    plaintiff = st.text_input("Plaintiff / Petitioner Name")
    defendant = st.text_input("Defendant / Respondent Name")
    description = st.text_area("Case Description", height=300, placeholder="Describe the facts of the case here...")
    
    predict_btn = st.button("🔮 Predict Verdict")

with col2:
    st.markdown("### 🧠 Agent Workflow & Results")
    
    if predict_btn:
        if not api_key:
            st.error("Please provide a Groq API Key in the sidebar.")
            st.stop()
            
        if not description:
            st.warning("Please enter a case description.")
            st.stop()

        status_container = st.empty()
        
        try:
            # Load Agents
            with st.spinner("Initializing Agents & Analytics Engine..."):
                router, parser, verdict_agent = load_reasoning_agents()
                retriever = load_retrieval_agent()
                analytics = load_analytics_engine()

            # --- EXECUTION PHASE ---
            
            # Step 1: Routing
            status_container.info("🔹 Agent 1 (Router): Analyzing case type...")
            category = router.route(description)
            # Removed Case Category Display as requested
            
            # Step 2: Parsing
            status_container.info("🔹 Agent 2 (Parser): Structuring case facts...")
            case_structure = parser.parse(description, plaintiff, defendant)

            # Step 3: Retrieval
            status_container.info(f"🔹 Agent 3 (Retrieval): Searching {category} database for precedents...")
            precedents = retriever.retrieve(case_structure['summary'], category)

            # Step 4: Verdict
            status_container.info("🔹 Agent 4 (Verdict): Generating prediction...")
            raw_verdict = verdict_agent.predict(case_structure, precedents, category)
            
            # Parse Outcome
            lines = raw_verdict.split('\n')
            outcome = "Unknown"
            verdict_text = raw_verdict
            
            if lines and lines[0].startswith("IN FAVOUR OF:"):
                outcome = lines[0].replace("IN FAVOUR OF:", "").strip()
                verdict_text = "\n".join(lines[1:]).strip()
            
            status_container.empty()

            # --- DISPLAY PHASE ---
            
            # 1. Probable Verdict (Top)
            st.markdown("### 🏛️ Probable Verdict")
            
            # Outcome Banner
            outcome_lower = outcome.lower()
            if "plaintiff" in outcome_lower or "petitioner" in outcome_lower:
                st.markdown(f'<div class="outcome-banner outcome-plaintiff">IN FAVOUR OF: {outcome.upper()}</div>', unsafe_allow_html=True)
            elif "defendant" in outcome_lower or "respondent" in outcome_lower:
                st.markdown(f'<div class="outcome-banner outcome-defendant">IN FAVOUR OF: {outcome.upper()}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="outcome-banner outcome-neutral">OUTCOME: {outcome.upper()}</div>', unsafe_allow_html=True)
            
            # Verdict Content
            st.markdown(f'<div class="verdict-card">{verdict_text}</div>', unsafe_allow_html=True)
            st.markdown("---")

            # 2. Structured Facts (Collapsible)
            with st.expander("View Structured Case Facts"):
                st.json(case_structure)

            # 3. Similar Precedents (Collapsible)
            st.write(f"**Found {len(precedents)} Similar Precedents:**")
            for i, p in enumerate(precedents, 1):
                with st.expander(f"{i}. {p.get('title', 'Unknown Case')}"):
                    st.write(f"**Verdict:** {p.get('verdict')}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())
