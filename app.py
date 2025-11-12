import streamlit as st
import pdfplumber
import docx
from textblob import TextBlob
import io
import re
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer

# --- Page Configuration (MUST be the first st command) ---
st.set_page_config(
    page_title="Advanced Feedback Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- MOCK DATABASE & KEYWORDS (to simulate entity linking) ---
MOCK_OFFICERS_DB = {
    "smith": {"id": 123, "name": "Officer Smith", "unit_id": 14},
    "doe": {"id": 456, "name": "Detective Doe", "unit_id": 14},
    "davis": {"id": 789, "name": "Sgt. Davis", "unit_id": 15},
}

MOCK_UNITS_DB = {
    14: "14th Precinct",
    15: "Traffic Division (K-9)",
}

MOCK_TOPIC_KEYWORDS = {
    "compassion": "community_engagement",
    "kind": "community_engagement",
    "helped the community": "community_engagement",
    "de-escalated": "de_escalation",
    "calmed the situation": "de_escalation",
    "fast response": "rapid_response",
    "arrived quickly": "rapid_response",
    "professional": "procedural_correctness",
    "by the book": "procedural_correctness",
}

# --- AI MODEL LOADING (with caching) ---
# @st.cache_resource is CRITICAL. It prevents reloading the large models
# every time the user interacts with the app.

@st.cache_resource
def load_translator(model_name):
    """Loads a specific translation model and tokenizer."""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading translation model {model_name}: {e}")
        return None, None

@st.cache_resource
def load_summarizer():
    """Loads the summarization pipeline."""
    try:
        # Using a smaller, faster model for Streamlit
        return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

@st.cache_resource
def load_qa_pipeline():
    """Loads the question-answering pipeline."""
    try:
        # Using a smaller, faster model
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"Error loading Q&A model: {e}")
        return None

# --- HELPER FUNCTIONS ---

def get_text_from_file(uploaded_file):
    """Extracts raw text from PDF, DOCX, or TXT files."""
    text = ""
    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None
    return text

def translate_to_english(text, lang):
    """Translates text to English if a model is available."""
    LANGUAGE_MODEL_MAP = {
        "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi to English
        "or": "Helsinki-NLP/opus-mt-or-en",  # Odia to English
        # Add more language codes and models here
    }
    
    if lang not in LANGUAGE_MODEL_MAP:
        st.warning(f"No translation model available for language '{lang}'. Processing in original language.")
        return text

    model_name = LANGUAGE_MODEL_MAP[lang]
    
    with st.spinner(f"Translating from '{lang}' to English..."):
        try:
            model, tokenizer = load_translator(model_name)
            if model is None:
                return text

            # Prepare text for translation
            input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
            
            # Generate translation
            translated_ids = model.generate(input_ids)
            
            # Decode and return
            translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return text

def process_text_pipeline(raw_text):
    """
    The main AI pipeline that orchestrates all tasks.
    """
    # --- 1. Language Detection & Translation ---
    processed_text = raw_text
    original_lang = "en"
    try:
        original_lang = detect(raw_text)
        if original_lang != "en":
            processed_text = translate_to_english(raw_text, original_lang)
    except Exception as e:
        st.warning(f"Language detection failed. Assuming English. Error: {e}")
        pass

    # --- 2. Dashboard Extraction ---
    # (Runs on the *translated* text)
    
    # NER (Entity Linking)
    found_officer = None
    officer_name = None
    for keyword, officer_data in MOCK_OFFICERS_DB.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', processed_text, re.IGNORECASE):
            found_officer = officer_data
            officer_name = officer_data['name']
            break

    suggested_officer_id = found_officer["id"] if found_officer else None
    suggested_unit_id = found_officer["unit_id"] if found_officer else None
    suggested_unit_name = MOCK_UNITS_DB.get(suggested_unit_id, "Unknown Unit") if suggested_unit_id else "N/A"

    # Topic Tagging
    suggested_tags = set()
    for keyword, tag in MOCK_TOPIC_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', processed_text, re.IGNORECASE):
            suggested_tags.add(tag)

    # Sentiment Analysis
    blob = TextBlob(processed_text)
    sentiment_score = blob.sentiment.polarity

    # Find Relevant Snippet
    extracted_text = "No relevant snippet found."
    if officer_name:
        sentences = re.split(r'[.!?]', processed_text)
        for sentence in sentences:
            if re.search(r'\b' + re.escape(officer_name.split()[-1]) + r'\b', sentence, re.IGNORECASE):
                extracted_text = sentence.strip() + "..."
                break
    elif len(processed_text) > 150:
         extracted_text = processed_text[:150].strip() + "..."
    elif processed_text:
        extracted_text = processed_text.strip()
    
    # --- 3. Summarization ---
    summary = "Summarization model failed or text too short."
    if len(processed_text.split()) > 30: # Only summarize if text is long enough
        try:
            summarizer = load_summarizer()
            if summarizer:
                summary_result = summarizer(processed_text, max_length=150, min_length=25, do_sample=False)
                summary = summary_result[0]['summary_text']
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            summary = "Summarization error."

    # --- Assemble Final Output ---
    output = {
        "original_text": raw_text,
        "processed_text": processed_text, # The English version
        "original_lang": original_lang,
        "summary": summary,
        "dashboard_details": {
            "extracted_text": extracted_text,
            "suggested_officer_id": suggested_officer_id,
            "suggested_unit_id": suggested_unit_id,
            "suggested_unit_name": suggested_unit_name,
            "suggested_sentiment": round(sentiment_score, 2),
            "suggested_tags": list(suggested_tags) or ["no_tags_found"]
        }
    }
    
    return output

# --- STREAMLIT APP UI ---

st.title("🤖 Advanced Feedback Analysis Pipeline")
st.markdown("This tool translates, analyzes, summarizes, and answers questions from community feedback.")

# --- Use session state to store results ---
# This is key for the Q&A to work without re-analyzing
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- Sidebar: Mock Data & Instructions ---
with st.sidebar:
    st.title("ℹ️ App Guide")
    st.markdown("**1. Input:** Upload a file or paste text (Hindi, Odia, or English).")
    st.markdown("**2. Analyze:** Click the 'Analyze' button.")
    st.markdown("**3. Review:** Check the 'Dashboard', 'Summary', and 'Q&A' tabs.")
    st.markdown("---")
    st.subheader("Mock Database (For Demo)")
    st.caption("The app links names to this data:")
    with st.expander("Officers DB", expanded=False):
        st.json(MOCK_OFFICERS_DB)
    with st.expander("Units DB", expanded=False):
        st.json(MOCK_UNITS_DB)
    with st.expander("Topic Keywords", expanded=False):
        st.json(MOCK_TOPIC_KEYWORDS)

# --- Main App Body ---

# Input Section
input_tab1, input_tab2 = st.tabs(["📁 Upload a File", "📋 Paste Text"])
raw_text_input = None
uploaded_file = None
pasted_text = None

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False
    )
    if uploaded_file:
        raw_text_input = get_text_from_file(uploaded_file)

with input_tab2:
    pasted_text = st.text_area("Paste your text here (e.g., from a news article or email):", height=250)
    if pasted_text:
        raw_text_input = pasted_text

# Analyze Button
if st.button("Analyze Text", type="primary", use_container_width=True):
    if raw_text_input:
        with st.spinner("🧠 Starting full AI pipeline... This may take a moment."):
            st.session_state.analysis_result = process_text_pipeline(raw_text_input)
    else:
        st.warning("Please upload a file or paste text first.")

# --- Output Section ---
if st.session_state.analysis_result:
    st.markdown("---")
    st.success("Analysis Complete!")
    
    result = st.session_state.analysis_result
    
    # Output Tabs
    out_tab1, out_tab2, out_tab3, out_tab4 = st.tabs([
        "📊 Dashboard Details", 
        "📝 Summary", 
        "❓ Ask a Question (Q&A)", 
        "📜 Original vs. Translated"
    ])
    
    # --- Tab 1: Dashboard Details ---
    with out_tab1:
        st.subheader("Data for Recognition Dashboard")
        
        details = result['dashboard_details']
        
        # Sentiment Metric
        sentiment = details['suggested_sentiment']
        if sentiment > 0.3:
            sentiment_label = "Positive"
            delta = f"{sentiment} (Good)"
        elif sentiment < -0.3:
            sentiment_label = "Negative"
            delta = f"{sentiment} (Bad)"
        else:
            sentiment_label = "Neutral"
            delta = f"{sentiment}"
        
        st.metric(label="Sentiment", value=sentiment_label, delta=delta)
        
        # Extracted Details
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Officer ID: {details['suggested_officer_id']}")
            st.text(f"Unit ID:    {details['suggested_unit_id']}")
            st.text(f"Unit Name:  {details['suggested_unit_name']}")
        
        with col2:
            st.text("Suggested Tags:")
            st.code(", ".join(details['suggested_tags']), language=None)
            
        st.subheader("Relevant Snippet")
        st.info(f"`{details['extracted_text']}`")
        
        st.subheader("Forward to Dashboard (Mock)")
        if st.button("Approve & Send to Dashboard"):
            st.success("Approved! (This is a demo - no data was sent)")

    # --- Tab 2: Summary ---
    with out_tab2:
        st.subheader("AI-Generated Summary")
        st.info(result['summary'])

    # --- Tab 3: Ask a Question ---
    with out_tab3:
        st.subheader("Ask a Question About the Text")
        
        # We use the 'processed_text' (English) as the context
        context = result['processed_text']
        
        question = st.text_input("Ask something like 'Who was the officer?' or 'What was the outcome?'")
        
        if question:
            with st.spinner("Finding answer..."):
                try:
                    qa_pipeline = load_qa_pipeline()
                    if qa_pipeline:
                        qa_result = qa_pipeline(question=question, context=context)
                        st.success(f"**Answer:** {qa_result['answer']}")
                        st.caption(f"(Confidence: {qa_result['score']:.2f})")
                    else:
                        st.error("Q&A model is not available.")
                except Exception as e:
                    st.error(f"Q&A failed: {e}")

    # --- Tab 4: Original vs. Translated ---
    with out_tab4:
        st.subheader("Text Processing")
        st.markdown(f"**Detected Language:** `{result['original_lang']}`")
        
        if result['original_lang'] != 'en':
            st.subheader("Original Text")
            st.text_area("", result['original_text'], height=200)
            
            st.subheader("Translated Text (Used for Analysis)")
            st.text_area("", result['processed_text'], height=200)
        else:
            st.info("Original text is in English. No translation needed.")
            st.text_area("Original Text", result['original_text'], height=200)