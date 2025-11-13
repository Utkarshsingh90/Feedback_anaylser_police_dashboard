"""
Police Recognition Analytics Dashboard with duplicate-file cleaner,
entity extraction, fuzzy gazetteer matching, sentiment, summary,
visual analytics, QA, per-item PDF summary and bulk export.

Place your datasets under ./data/
 - OdishaIPCCrimedata.json
 - DistrictReport.json
 - mock_cctnsdata.json
 - publicfeedback.json

Run:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    streamlit run app.py
"""

import streamlit as st
import os
import json
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple, Optional
import pdfplumber
import re
from datetime import datetime
from io import BytesIO
import pandas as pd
import altair as alt
from rapidfuzz import process as rf_process, fuzz
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from dateutil import parser as dateparser
import base64

st.set_page_config(page_title="Police Recognition Analytics", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Config & constants
# ---------------------------
DATA_DIR = Path("data")
EXPECTED_FILES = [
    "OdishaIPCCrimedata.json",
    "DistrictReport.json",
    "mock_cctnsdata.json",
    "publicfeedback.json"
]

# fallback small gazetteer if none provided
FALLBACK_GAZETTEER = {
    "names": [
        "Officer John Smith", "Sergeant Mary Johnson", "Inspector David Brown",
        "Captain Sarah Williams", "Detective Michael Jones", "Officer Emily Davis",
        "Lieutenant Robert Miller", "Chief Patricia Wilson", "Officer James Moore",
        "Constable Jennifer Taylor", "SI Rajesh Kumar", "ASI Priya Sharma",
        "Inspector Amit Patel", "Constable Sunita Verma", "Officer Rahul Singh"
    ],
    "departments": [
        "Central Police Station", "North District Police", "South Precinct",
        "Crime Investigation Department", "Traffic Police Department", "City Police Commissionerate",
        "Bhubaneswar Police", "Cuttack Police", "Puri Police Station"
    ],
    "locations": [
        "Bhubaneswar", "Cuttack", "Puri", "Kendrapara", "Bhubaneshwar", "Downtown", "Riverside", "City Center"
    ]
}

TITLE_REGEX = r'\b(Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lieutenant|SI|ASI|Sub-Inspector|SP|DSP|ACP|DCP)\b\.?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'

CRIME_KEYWORDS = [
    "murder", "rape", "kidnapping", "abduction", "dacoity", "robbery",
    "burglary", "theft", "riots", "cheating", "counterfeiting", "arson",
    "hurt", "dowry", "assault", "cruelty", "importation", "negligence"
]

# ---------------------------
# Utility functions
# ---------------------------
def list_data_files() -> List[Path]:
    DATA_DIR.mkdir(exist_ok=True)
    return sorted([p for p in DATA_DIR.iterdir() if p.is_file()])

def load_json_safe(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def file_hash(p: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def scan_duplicates_by_hash(files: List[Path]) -> Dict[str, List[Path]]:
    mapping = {}
    for p in files:
        h = file_hash(p)
        mapping.setdefault(h, []).append(p)
    # only keep entries with more than one file (duplicates)
    return {h: ps for h, ps in mapping.items() if len(ps) > 1}

def scan_duplicates_by_name_pattern(files: List[Path], patterns: List[str] = None) -> List[Path]:
    patterns = patterns or ["(1)", "copy", "duplicate", "- Copy"]
    dup_files = []
    for p in files:
        name = p.name.lower()
        if any(pat.lower() in name for pat in patterns):
            dup_files.append(p)
    return dup_files

def delete_files(paths: List[Path]) -> List[str]:
    deleted = []
    for p in paths:
        try:
            p.unlink()
            deleted.append(p.name)
        except Exception as e:
            st.error(f"Failed to delete {p.name}: {e}")
    return deleted

# ---------------------------
# Text extraction & NLP
# ---------------------------
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

def extract_text_from_pdf_file(uploaded_file) -> str:
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text += ptext + "\n"
    except Exception:
        # fallback: return empty
        return ""
    return text.strip()

def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    persons, orgs, locs = [], [], []
    for ent in doc.ents:
        label = ent.label_
        if label == "PERSON":
            persons.append(ent.text.strip())
        elif label in ("ORG", "NORP"):
            orgs.append(ent.text.strip())
        elif label in ("GPE", "LOC", "FAC"):
            locs.append(ent.text.strip())
    # dedupe preserving order
    def dedupe(seq):
        seen = set(); out = []
        for s in seq:
            if s not in seen:
                seen.add(s); out.append(s)
        return out
    return {"persons": dedupe(persons), "orgs": dedupe(orgs), "locs": dedupe(locs)}

def extract_ranked_names(text: str) -> List[str]:
    matches = []
    for m in re.finditer(TITLE_REGEX, text):
        title = m.group(1)
        name = m.group(2)
        matches.append(f"{title} {name}")
    return matches

def fuzzy_match(candidate: str, pool: List[str], cutoff: int = 70) -> Optional[Tuple[str,int]]:
    if not pool:
        return None
    best = rf_process.extractOne(candidate, pool, scorer=fuzz.WRatio)
    if best and best[1] >= cutoff:
        return (best[0], int(best[1]))
    return None

def extract_crime_tags(text: str) -> List[str]:
    t = text.lower()
    found = []
    for k in CRIME_KEYWORDS:
        if k in t:
            found.append(k)
    # sections if any
    sec_matches = re.findall(r'Section\s*[:\-]?\s*([0-9]{2,4})', text, flags=re.IGNORECASE)
    for s in sec_matches:
        found.append(f"Section {s}")
    return list(dict.fromkeys(found))

def extract_dates(text: str) -> List[str]:
    patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
    ]
    found = set()
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            try:
                dt = dateparser.parse(m, fuzzy=True)
                if dt:
                    found.add(dt.date().isoformat())
                else:
                    found.add(m)
            except:
                found.add(m)
    # fuzzy date phrases
    phrases = re.findall(r'\b(on|dated|on date)\s+([A-Za-z0-9,\-\/\s]{3,30})', text, flags=re.IGNORECASE)
    for _, candidate in phrases:
        try:
            dt = dateparser.parse(candidate, fuzzy=True)
            if dt:
                found.add(dt.date().isoformat())
        except:
            pass
    return sorted(found)

def summarize_text_simple(text: str, max_sentences: int = 3) -> str:
    # simple extractive: first few sentences of reasonable length
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
    if not sents:
        return text[:500]
    return " ".join(sents[:max_sentences])

def sentiment_score(text: str) -> Tuple[str, float]:
    v = sentiment_analyzer.polarity_scores(text[:1000])
    compound = v["compound"]
    label = "NEUTRAL"
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    return label, compound

# ---------------------------
# Gazetteer & hybrid extraction
# ---------------------------
def build_gazetteer_from_data(datasets: Dict[str, Optional[object]]) -> Dict[str, List[str]]:
    gaz = {"names": [], "departments": [], "locations": []}
    # mock_cctnsdata -> extract officer_name and rank
    mc = datasets.get("mock_cctnsdata")
    if isinstance(mc, list):
        for r in mc:
            n = r.get("officer_name")
            if n:
                gaz["names"].append(n)
            idf = r.get("id") or r.get("fir_no")
            if idf and str(idf) not in gaz["names"]:
                # not a name, skip
                pass
    # district reports may have police_station and investigating_officer_id
    dr = datasets.get("district_reports")
    if isinstance(dr, list):
        for r in dr:
            ps = r.get("police_station") or r.get("station") or r.get("police_station_name")
            if ps:
                gaz["departments"].append(ps)
            io = r.get("investigating_officer_id")
            if io:
                gaz["names"].append(io)
    # public feedback may contain names in free text; we won't parse here
    # odisha ipc data may include district names or other keys
    od = datasets.get("odisha_ipc")
    if isinstance(od, dict):
        for k in od.keys():
            # some districts as keys
            gaz["locations"].append(k)
    # always fallback to built-in small gazetteer
    for k in ("names", "departments", "locations"):
        gaz[k] = list(dict.fromkeys(gaz.get(k, []) + FALLBACK_GAZETTEER.get(k, [])))
    return gaz

def hybrid_extract(text: str, gazetteer: Dict[str, List[str]]) -> Dict:
    entities = extract_entities_spacy(text)
    ranked = extract_ranked_names(text)
    candidates = []
    # prefer ranked names then NER persons
    candidates.extend(ranked)
    candidates.extend(entities.get("persons", []))
    found_officers = []
    found_officers_conf = []
    for cand in candidates:
        # try exact match
        matched = None
        for opt in gazetteer["names"]:
            if cand.lower() == opt.lower():
                matched = (opt, 100)
                break
        if not matched:
            fr = fuzzy_match(cand, gazetteer["names"], cutoff=70)
            if fr:
                matched = fr
        if matched:
            found_officers.append(matched[0])
            found_officers_conf.append({"name": matched[0], "score": matched[1], "source": "gazetteer/ner"})
        else:
            # keep raw candidate with lower confidence
            found_officers.append(cand)
            found_officers_conf.append({"name": cand, "score": 55, "source": "ner/raw"})
    # de-dup
    found_officers = list(dict.fromkeys(found_officers)) or ["(Officer name not found - please specify)"]
    # departments
    found_depts = []
    for org in entities.get("orgs", []):
        f = fuzzy_match(org, gazetteer["departments"], cutoff=60)
        if f:
            found_depts.append(f[0])
        else:
            found_depts.append(org)
    # fallback: find department keywords in text
    if not found_depts:
        for dept in gazetteer["departments"]:
            if dept.lower() in text.lower():
                found_depts.append(dept)
    found_depts = list(dict.fromkeys(found_depts)) or ["(Department not specified)"]
    # locations
    found_locs = []
    for loc in entities.get("locs", []):
        f = fuzzy_match(loc, gazetteer["locations"], cutoff=65)
        if f:
            found_locs.append(f[0])
        else:
            found_locs.append(loc)
    if not found_locs:
        for loc in gazetteer["locations"]:
            if loc.lower() in text.lower():
                found_locs.append(loc)
    found_locs = list(dict.fromkeys(found_locs))
    return {
        "officers": found_officers,
        "officers_conf": found_officers_conf,
        "departments": found_depts,
        "locations": found_locs
    }

# ---------------------------
# PDF report creation
# ---------------------------
def create_pdf_from_result(result: Dict) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1f4b99"), alignment=1)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#2e6bb7"))
    story.append(Paragraph("Police Recognition Report", title_style))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(result.get("summary", "(no summary)"), styles['Normal']))
    story.append(Spacer(1, 0.12*inch))
    metrics = [
        ["Metric", "Value"],
        ["Recognition Score", f"{result.get('recognition_score', 0)}/1.0"],
        ["Sentiment", result.get("sentiment_label", "NEUTRAL")],
        ["Text length", str(result.get("text_length", 0))],
        ["Crime tags", ", ".join(result.get("crime_tags", []) or ["None"])],
        ["Dates", ", ".join(result.get("dates", []) or ["None"])]
    ]
    t = Table(metrics, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#2e6bb7")),
                           ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                           ('GRID',(0,0),(-1,-1),1,colors.black),
                           ('BACKGROUND',(0,1),(-1,-1),colors.beige)]))
    story.append(t)
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Identified Officers", heading_style))
    for o in result.get("officers", []):
        story.append(Paragraph(f"• {o}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Departments", heading_style))
    for d in result.get("departments", []):
        story.append(Paragraph(f"• {d}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Locations", heading_style))
    for l in result.get("locations", []):
        story.append(Paragraph(f"• {l}", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------------
# Processing pipeline
# ---------------------------
def analyze_text(text: str, datasets: Dict[str, Optional[object]]) -> Dict:
    text = text.strip()
    gaz = build_gazetteer_from_data(datasets)
    hybrid = hybrid_extract(text, gaz)
    # sentiment
    sent_label, sent_score = sentiment_score(text)
    # summary
    summary = summarize_text_simple(text)
    # crime tags and dates
    crimes = extract_crime_tags(text)
    dates = extract_dates(text)
    # recognition score heuristic
    base = (sent_score + 1) / 2  # normalize -1..1 to 0..1
    tag_boost = 0.0
    if any(w in text.lower() for w in ["saved", "rescue", "life-saving", "revived"]):
        tag_boost += 0.12
    if any(w in text.lower() for w in ["brave", "courage", "heroic", "fearless"]):
        tag_boost += 0.12
    length_boost = min(0.1, len(text)/1000 * 0.1)
    recognition_score = round(min(1.0, base + tag_boost + length_boost), 3)
    result = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "summary": summary,
        "officers": hybrid.get("officers", []),
        "officers_conf": hybrid.get("officers_conf", []),
        "departments": hybrid.get("departments", []),
        "locations": hybrid.get("locations", []),
        "sentiment_label": sent_label,
        "sentiment_score": sent_score,
        "crime_tags": crimes,
        "dates": dates,
        "recognition_score": recognition_score,
        "text_length": len(text)
    }
    return result

# ---------------------------
# Session state
# ---------------------------
if "processed_data" not in st.session_state:
    st.session_state.processed_data = []

if "datasets" not in st.session_state:
    # load datasets into memory once
    files = {p.name: load_json_safe(p) for p in list_data_files()}
    # normalize keys
    st.session_state.datasets = {
        "odisha_ipc": files.get("OdishaIPCCrimedata.json"),
        "district_reports": files.get("DistrictReport.json") or files.get("DistrictReport.JSON"),
        "mock_cctnsdata": files.get("mock_cctnsdata.json"),
        "publicfeedback": files.get("publicfeedback.json")
    }

# ---------------------------
# UI layout
# ---------------------------
st.title("🚔 Police Recognition Analytics")
st.markdown("Use the panels to clean data files, upload documents, analyze feedback, visualize results, and export reports.")

# Sidebar: file status + duplicate cleaner
with st.sidebar:
    st.header("Data files")
    data_files = list_data_files()
    file_table = []
    for p in data_files:
        status = "✅" if p.name in EXPECTED_FILES else "•"
        file_table.append((p.name, p.stat().st_size, status))
    if file_table:
        df_files = pd.DataFrame(file_table, columns=["Filename", "Size (bytes)", "Note"])
        st.table(df_files)
    else:
        st.info("No files in data/ folder yet.")

    st.markdown("---")
    st.subheader("Duplicate file cleaner")
    if st.button("Scan duplicates"):
        files = list_data_files()
        dup_hash = scan_duplicates_by_hash(files)
        dup_name = scan_duplicates_by_name_pattern(files)
        st.session_state.dup_by_hash = dup_hash
        st.session_state.dup_by_name = dup_name
        if not dup_hash and not dup_name:
            st.success("No obvious duplicates found.")
        else:
            st.warning("Duplicates detected — review below and delete selected items carefully.")

    if "dup_by_hash" in st.session_state and st.session_state.dup_by_hash:
        st.markdown("### Duplicates (by identical content)")
        for h, paths in st.session_state.dup_by_hash.items():
            st.markdown(f"- **Group hash:** `{h[:10]}...` (same content)")
            for p in paths:
                st.checkbox(f"{p}", key=f"chk_hash_{p.name}")
        if st.button("Delete selected identical files"):
            to_delete = []
            for h, paths in st.session_state.dup_by_hash.items():
                for p in paths:
                    if st.session_state.get(f"chk_hash_{p.name}", False):
                        to_delete.append(p)
            if to_delete:
                deleted = delete_files(to_delete)
                st.success(f"Deleted: {deleted}")
                # refresh dataset cache
                st.session_state.datasets = {k: load_json_safe(DATA_DIR / k) for k in EXPECTED_FILES}
            else:
                st.info("No files selected for deletion.")
    if "dup_by_name" in st.session_state and st.session_state.dup_by_name:
        st.markdown("### Duplicates (by name pattern)")
        for p in st.session_state.dup_by_name:
            st.checkbox(f"{p}", key=f"chk_name_{p.name}")
        if st.button("Delete selected name-pattern files"):
            to_delete = [p for p in st.session_state.dup_by_name if st.session_state.get(f"chk_name_{p.name}", False)]
            if to_delete:
                deleted = delete_files(to_delete)
                st.success(f"Deleted: {deleted}")
                st.session_state.datasets = {k: load_json_safe(DATA_DIR / k) for k in EXPECTED_FILES}
            else:
                st.info("No files selected for deletion.")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Process Feedback", "Dashboard", "Q&A", "Export & Files"])

# ---------------------------
# Tab1 - Process Feedback
# ---------------------------
with tab1:
    st.header("Process feedback or reports")
    col1, col2 = st.columns([3,1])
    with col1:
        input_choice = st.radio("Input method", ("Paste text", "Upload file (txt/pdf)"))
        raw_text = ""
        if input_choice == "Paste text":
            raw_text = st.text_area("Paste the text to analyze", height=280, placeholder="Type or paste report / feedback here...")
        else:
            uploaded = st.file_uploader("Upload TXT or PDF", type=["txt","pdf"])
            if uploaded:
                if uploaded.type == "application/pdf":
                    try:
                        raw_text = extract_text_from_pdf_file(uploaded)
                        st.success(f"Extracted {len(raw_text)} characters from PDF")
                    except Exception as e:
                        st.error(f"PDF read error: {e}")
                else:
                    try:
                        raw_text = uploaded.read().decode("utf-8", errors="ignore")
                        st.success(f"Loaded {len(raw_text)} characters from TXT")
                    except Exception as e:
                        st.error(f"Text read error: {e}")
        if st.button("Analyze"):
            if not raw_text or not raw_text.strip():
                st.warning("Please provide text or upload a file.")
            else:
                result = analyze_text(raw_text, st.session_state.datasets)
                st.session_state.processed_data.append(result)
                st.success("Analysis complete — item added to processed data.")
                st.experimental_rerun()
    with col2:
        st.markdown("### Tips")
        st.markdown("- Include rank/title (e.g., `SI Rajesh Kumar`) when available for better extraction.")
        st.markdown("- For best PDF results, upload text-based PDFs; scanned PDFs need OCR before uploading.")
        st.markdown("- Use the duplicate cleaner in the sidebar to remove accidental uploads.")

# ---------------------------
# Tab2 - Dashboard (visual analytics)
# ---------------------------
with tab2:
    st.header("Visual analytics")
    data = st.session_state.processed_data
    if not data:
        st.info("No processed items yet. Analyze some feedback to build visuals.")
    else:
        df = pd.DataFrame(data)
        # crime type distribution
        all_crimes = [c for row in df['crime_tags'] for c in (row or [])]
        crime_series = pd.Series(all_crimes).value_counts().reset_index()
        crime_series.columns = ["crime", "count"]
        if not crime_series.empty:
            st.subheader("Crime type distribution")
            chart = alt.Chart(crime_series).mark_bar().encode(
                x=alt.X("crime:N", sort='-y', title="Crime Type"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["crime","count"]
            ).properties(width=700, height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No crime tags found in processed items.")

        # district-wise counts from datasets if available
        st.subheader("District / Station counts (from DistrictReport dataset if present)")
        dr = st.session_state.datasets.get("DistrictReport.json") or st.session_state.datasets.get("DistrictReport.JSON") or st.session_state.datasets.get("DistrictReport.json".lower())
        if isinstance(dr, list) and dr:
            # attempt to extract district/station and counts fields
            rows = []
            for r in dr:
                district = r.get("district") or r.get("District") or r.get("district_name") or r.get("station")
                cases = r.get("cases") or r.get("cases_count") or r.get("casesReported") or r.get("cases", 0)
                try:
                    cases = int(cases)
                except:
                    cases = 0
                rows.append({"district": district or "Unknown", "cases": cases})
            df_dr = pd.DataFrame(rows).groupby("district", as_index=False).sum()
            bar = alt.Chart(df_dr).mark_bar().encode(
                x=alt.X("district:N", sort='-y', title="District/Station"),
                y=alt.Y("cases:Q", title="Cases"),
                tooltip=["district","cases"]
            ).properties(width=700, height=350)
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("District report dataset not present or empty.")

        # sentiment distribution
        st.subheader("Sentiment distribution")
        sent_counts = df['sentiment_label'].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        if not sent_counts.empty:
            pie = alt.Chart(sent_counts).mark_arc().encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="sentiment", type="nominal"),
                tooltip=["sentiment","count"]
            ).properties(width=400, height=300)
            st.altair_chart(pie, use_container_width=False)
        # top officers
        st.subheader("Top mentioned officers")
        all_offs = [o for row in df['officers'] for o in (row or []) if not str(o).startswith("(")]
        if all_offs:
            off_series = pd.Series(all_offs).value_counts().reset_index()
            off_series.columns = ["officer", "count"]
            bar2 = alt.Chart(off_series.head(15)).mark_bar().encode(
                x=alt.X("officer:N", sort='-y', title="Officer"),
                y=alt.Y("count:Q", title="Mentions"),
                tooltip=["officer", "count"]
            ).properties(width=700, height=350)
            st.altair_chart(bar2, use_container_width=True)
        else:
            st.info("No officer mentions found yet.")

# ---------------------------
# Tab3 - Q&A (keyword search over processed items)
# ---------------------------
with tab3:
    st.header("Q&A / Search over processed feedback")
    if not st.session_state.processed_data:
        st.info("No processed items yet.")
    else:
        q = st.text_input("Enter a question or keywords (e.g., 'bravery', 'stadium fire', 'SI Rajesh'):")
        top_k = st.slider("Max results", 1, 10, 5)
        if st.button("Search"):
            results = []
            for item in st.session_state.processed_data:
                text = item.get("text","")
                score = fuzz.partial_ratio(q.lower(), text.lower())
                if score > 30:
                    results.append((score, item))
            results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
            if results:
                for sc, it in results:
                    st.markdown(f"**Score:** {sc}  |  **Recognition Score:** {it.get('recognition_score')}")
                    st.write(it.get("summary"))
                    st.write("**Officers:**", ", ".join(it.get("officers", [])))
                    if st.button(f"Download PDF for result {sc}", key=f"pdf_{sc}_{len(it.get('summary',''))}"):
                        pdfb = create_pdf_from_result({
                            "summary": it.get("summary"),
                            "recognition_score": it.get("recognition_score"),
                            "sentiment_label": it.get("sentiment_label"),
                            "text_length": it.get("text_length"),
                            "crime_tags": it.get("crime_tags"),
                            "dates": it.get("dates"),
                            "officers": it.get("officers"),
                            "departments": it.get("departments"),
                            "locations": it.get("locations")
                        })
                        st.download_button("Download report PDF", data=pdfb, file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                st.info("No matches found. Try different keywords.")

# ---------------------------
# Tab4 - Export & Files
# ---------------------------
with tab4:
    st.header("Export & Files")
    st.subheader("Processed items")
    if not st.session_state.processed_data:
        st.info("No processed items to export.")
    else:
        df_proc = pd.DataFrame(st.session_state.processed_data)
        st.dataframe(df_proc, height=400)
        csv_data = df_proc.to_csv(index=False)
        json_data = df_proc.to_json(orient="records", indent=2, force_ascii=False)
        st.download_button("Download CSV", csv_data, file_name=f"processed_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        st.download_button("Download JSON", json_data, file_name=f"processed_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")
    st.markdown("---")
    st.subheader("Data folder quick actions")
    if st.button("Refresh datasets"):
        # reload dataset cache
        st.session_state.datasets = {k: load_json_safe(DATA_DIR / k) for k in EXPECTED_FILES}
        st.success("Datasets refreshed.")
    st.info("Files listed in the sidebar show current contents of the data/ folder. Use the duplicate cleaner above to remove accidental uploads.")
