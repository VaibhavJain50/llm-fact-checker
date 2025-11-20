import streamlit as st
from fact_checker import FactChecker

st.set_page_config(page_title="Fact Checker", page_icon="🔍")

st.title("LLM Fact Checker (Powered by Gemini)")
st.markdown("### RAG Pipeline: `spaCy` + `ChromaDB` + `Gemini Flash`")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google API Key", type="password")
    st.caption("Get a free key from AI Studio (aistudio.google.com)")
    st.markdown("---")
    st.info("Model: **Gemini 2.5 Flash**\n\nChosen for low latency and high efficiency.")

if api_key:
    # Initialize Logic
    @st.cache_resource
    def get_checker(key):
        checker = FactChecker(key)
        checker.load_knowledge_base("facts.csv")
        return checker

    checker = get_checker(api_key)
    
    # Main Input
    claim = st.text_area("Claim to Verify:", 
                         value="The Indian government has announced free electricity to all farmers starting July 2025.")
    
    if st.button("Check Claim"):
        with st.spinner("🤖 Consulting Knowledge Base..."):
            result = checker.check_claim(claim)
            
        # Display Results
        st.divider()
        
        # Verdict Badge
        verdict = result.get('verdict', 'Error')
        if verdict == "True":
            st.success(f"✅ VERDICT: {verdict}")
        elif verdict == "False":
            st.error(f"❌ VERDICT: {verdict}")
        else:
            st.warning(f"🤷 VERDICT: {verdict}")

        st.write(f"**Reasoning:** {result.get('reasoning')}")

        # Developer View (The "Entry Level" Bonus)
        with st.expander("See Under the Hood (Debug Info)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🗂 Retrieved Facts (Context)**")
                for f in result.get('retrieved_context', []):
                    st.caption(f"- {f}")
            
            with col2:
                st.markdown("**📊 Metadata**")
                st.text(f"Entities: {result.get('entities_detected')}")
                st.text(f"Similarity Scores: {result.get('confidence_scores')}")

else:
    st.warning("👈 Please enter your Google API Key to begin.")