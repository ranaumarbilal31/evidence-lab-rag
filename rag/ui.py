"""Presentation only: static, trusted markup. Evidence always uses plain text."""
import streamlit as st


def theme():
    st.html('''<style>
    :root { --ink:#173d3a; --muted:#59716e; --line:#dce6e2; --teal:#18766a; }
    .stApp { background:#f4f6f2; }
    .block-container { max-width:1320px; padding:4.5rem 3rem 3rem; }
    [data-testid="stSidebar"] { background:#ecf1eb; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding:2rem 1.25rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.045em; }
    h1 { font-size:3.3rem!important; line-height:1.07!important; max-width:780px; padding-top:.4rem!important; }
    h2 { font-size:1.5rem!important; } h3 { font-size:1.15rem!important; }
    p, label { color:#304d49; }
    [data-testid="stCaptionContainer"] p { color:var(--muted); }
    [data-testid="stText"] { white-space:pre-wrap; overflow-wrap:anywhere; font-family:inherit; line-height:1.7; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:16px!important; }
    .st-key-workspace, .st-key-results, .st-key-saved { background:#fff; border-radius:18px; }
    .stButton button, .stDownloadButton button { border-radius:9px; min-height:42px; font-weight:600; }
    .stButton button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
    .stButton button:focus-visible { outline:3px solid #d4a550; outline-offset:2px; }
    [data-testid="stTextInput"] input { min-height:48px; }
    [data-testid="stExpander"] { border-color:var(--line); border-radius:10px; }
    [data-testid="stAlert"] { border-radius:10px; }
    .lab-wordmark { font-size:1.3rem; font-weight:750; color:var(--ink); letter-spacing:-.04em; }
    .lab-mark { display:inline-block; padding:.15rem .55rem; margin-right:.4rem; background:#17675f; color:#fff; border-radius:8px; }
    .lab-eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#54766e; }
    .lab-masthead { display:flex; justify-content:space-between; align-items:center; padding-bottom:1.25rem; border-bottom:1px solid var(--line); margin-bottom:1.9rem; gap:1rem; }
    .lab-badge { padding:.35rem .75rem; border:1px solid #cedcd4; border-radius:30px; font-size:.75rem; color:#42675c; background:#edf3e9; }
    .lab-intro { max-width:700px; color:#59716e; font-size:1.05rem; line-height:1.7; margin:.4rem 0 1.4rem; }
    .lab-path { display:flex; gap:.7rem; align-items:center; color:#58736a; font-size:.8rem; margin:1rem 0 2rem; flex-wrap:wrap; }
    .lab-path span { padding:.5rem .8rem; background:#eaf0e7; border-radius:6px; }
    .lab-nav { padding:.8rem 1rem; border-radius:8px; background:#dce9dc; color:#24574d; font-size:.9rem; font-weight:600; margin:1.8rem 0; }
    .lab-note { font-size:.8rem; line-height:1.7; color:#567067; margin:1rem 0; }
    .lab-footer { border-top:1px solid var(--line); padding-top:1rem; margin-top:2rem; font-size:.75rem; color:#64796c; }
    @media(max-width:768px) {
      .block-container { padding:4.3rem 1rem 2rem; }
      h1 { font-size:2.25rem!important; }
      .lab-masthead { margin-bottom:1.3rem; }
      [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width:100%!important; flex:1 1 100%!important; min-width:0!important; }
    }
    @media(prefers-reduced-motion:reduce) { * { transition:none!important; animation:none!important; } }
    </style>''')


def header():
    st.html('''<div class="lab-masthead"><div class="lab-eyebrow">Evidence Lab / Research workspace</div><span class="lab-badge">RAG · Safety · Recovery</span></div>
    <div class="lab-eyebrow">Trust the evidence. Inspect the answer.</div>''')
    st.title('Better answers start with better evidence.')
    st.html('''<p class="lab-intro">Put ordinary retrieval and a Safety Wall side by side.
    Explore where an answer comes from, what gets blocked, and what can be recovered.</p>
    <div class="lab-path"><span>01 &nbsp; Choose evidence</span>→<span>02 &nbsp; Ask & compare</span>→<span>03 &nbsp; Inspect the sources</span></div>''')


def sidebar_brand():
    st.html('''<div class="lab-wordmark"><span class="lab-mark">◈</span> Evidence Lab</div>
    <div class="lab-note">A small experiment in<br>more trustworthy answers.</div>
    <div class="lab-nav">◈ &nbsp; Evidence workspace</div>''')
