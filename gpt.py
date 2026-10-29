import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="Your Hazu",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------- session state
# Each visitor gets their own isolated, private session state in memory.
# Keys are NEVER stored on the server disk or shared across users.
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_role" not in st.session_state:
    st.session_state.system_role = ""
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Auto (System)"

# ------------------------------------------------------------------ styling
def get_custom_css(theme_mode: str) -> str:
    light_vars = """
        --app-bg: linear-gradient(135deg, #f0f4fd 0%, #e5ecfb 45%, #f3effc 100%);
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --card-bg: rgba(255, 255, 255, 0.88);
        --card-border: rgba(255, 255, 255, 0.95);
        --card-shadow: 0 16px 36px -8px rgba(30, 41, 59, 0.12), 0 4px 12px rgba(0, 0, 0, 0.04), inset 0 1px 1px rgba(255, 255, 255, 1);
        --card-shadow-hover: 0 24px 48px -8px rgba(99, 102, 241, 0.22), 0 8px 18px rgba(0, 0, 0, 0.06), inset 0 1px 2px rgba(255, 255, 255, 1);
        --sidebar-bg: rgba(246, 248, 253, 0.92);
        --sidebar-border: rgba(203, 213, 225, 0.7);
        --memory-bg: rgba(255, 255, 255, 0.92);
        --memory-border: rgba(226, 232, 240, 0.95);
        --memory-shadow: 0 4px 14px rgba(0, 0, 0, 0.05), inset 0 1px 0 rgba(255, 255, 255, 1);
        --memory-label: #4f46e5;
        --user-msg-bg: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        --user-msg-text: #ffffff;
        --user-msg-shadow: 0 8px 22px rgba(99, 102, 241, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.4);
        --ai-msg-bg: rgba(255, 255, 255, 0.9);
        --ai-msg-border: rgba(226, 232, 240, 0.95);
        --ai-msg-text: #0f172a;
        --ai-msg-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 1);
        --btn-primary-bg: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        --btn-primary-text: #ffffff;
        --btn-primary-shadow: 0 6px 18px rgba(99, 102, 241, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.4);
        --btn-primary-hover: 0 10px 25px rgba(99, 102, 241, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.5);
        --input-bg: rgba(255, 255, 255, 0.95);
        --input-border: rgba(203, 213, 225, 0.9);
        --input-text: #0f172a;
        --input-shadow: 0 4px 14px rgba(0, 0, 0, 0.04), inset 0 1px 2px rgba(0, 0, 0, 0.02);
        --badge-bg: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --header-bg: linear-gradient(135deg, #3730a3 0%, #4f46e5 35%, #7c3aed 70%, #d946ef 100%);
        --header-shadow: 0 16px 36px -6px rgba(79, 70, 229, 0.35), inset 0 1px 2px rgba(255, 255, 255, 0.4);
    """

    dark_vars = """
        --app-bg: linear-gradient(135deg, #090a14 0%, #111428 50%, #191535 100%);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --card-bg: rgba(23, 26, 46, 0.75);
        --card-border: rgba(255, 255, 255, 0.12);
        --card-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.65), 0 0 25px rgba(124, 58, 237, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        --card-shadow-hover: 0 26px 60px -8px rgba(0, 0, 0, 0.75), 0 0 35px rgba(124, 58, 237, 0.3), inset 0 1px 2px rgba(255, 255, 255, 0.25);
        --sidebar-bg: rgba(13, 15, 27, 0.9);
        --sidebar-border: rgba(255, 255, 255, 0.08);
        --memory-bg: rgba(255, 255, 255, 0.06);
        --memory-border: rgba(255, 255, 255, 0.08);
        --memory-shadow: 0 6px 16px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        --memory-label: #a78bfa;
        --user-msg-bg: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        --user-msg-text: #ffffff;
        --user-msg-shadow: 0 8px 24px rgba(124, 58, 237, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25);
        --ai-msg-bg: rgba(23, 26, 46, 0.8);
        --ai-msg-border: rgba(255, 255, 255, 0.1);
        --ai-msg-text: #f1f5f9;
        --ai-msg-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        --btn-primary-bg: linear-gradient(135deg, #4f46e5 0%, #8b5cf6 100%);
        --btn-primary-text: #ffffff;
        --btn-primary-shadow: 0 8px 25px rgba(99, 102, 241, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        --btn-primary-hover: 0 12px 30px rgba(139, 92, 246, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.45);
        --input-bg: rgba(17, 19, 36, 0.88);
        --input-border: rgba(255, 255, 255, 0.14);
        --input-text: #f8fafc;
        --input-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 2px rgba(0, 0, 0, 0.2);
        --badge-bg: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --header-bg: linear-gradient(135deg, #312e81 0%, #4f46e5 35%, #8b5cf6 70%, #db2777 100%);
        --header-shadow: 0 18px 45px -8px rgba(124, 58, 237, 0.5), inset 0 1px 2px rgba(255, 255, 255, 0.25);
    """

    if theme_mode == "🌙 Dark Mode":
        theme_block = f":root, .stApp {{ {dark_vars} }}"
    elif theme_mode == "☀️ Light Mode":
        theme_block = f":root, .stApp {{ {light_vars} }}"
    else:  # Auto (System)
        theme_block = f"""
            :root, .stApp {{ {light_vars} }}
            @media (prefers-color-scheme: dark) {{
                :root, .stApp {{ {dark_vars} }}
            }}
        """

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    {theme_block}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: var(--app-bg) !important;
        color: var(--text-primary) !important;
        transition: background 0.3s ease, color 0.3s ease;
    }}

    p, span, label, h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
    }}

    /* 3D Glassmorphic Key Card */
    .key-card {{
        background: var(--card-bg);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid var(--card-border);
        border-radius: 26px;
        padding: 44px 38px;
        text-align: center;
        box-shadow: var(--card-shadow);
        transform: perspective(1000px) translateZ(0);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 24px;
    }}
    .key-card:hover {{
        box-shadow: var(--card-shadow-hover);
        transform: perspective(1000px) translateY(-3px) translateZ(6px);
    }}
    .key-card h1 {{
        font-family: 'Outfit', sans-serif;
        font-size: 46px;
        font-weight: 800;
        margin: 12px 0 6px 0;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .key-card p {{
        color: var(--text-secondary);
        font-size: 15px;
        line-height: 1.6;
    }}
    .key-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--badge-bg);
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.3);
    }}

    /* 3D Floating Header Banner */
    .hazu-header {{
        background: var(--header-bg);
        padding: 24px 32px;
        border-radius: 22px;
        box-shadow: var(--header-shadow);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.25);
        transform: perspective(1000px) translateZ(0);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .hazu-header:hover {{
        transform: translateY(-2px);
    }}
    .hazu-header::before {{
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .hazu-header h1 {{
        font-family: 'Outfit', sans-serif;
        color: #ffffff !important;
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    .hazu-header p {{
        color: rgba(255, 255, 255, 0.92) !important;
        margin: 6px 0 0 0;
        font-size: 14.5px;
        font-weight: 500;
    }}

    /* Sidebar Glass & 3D styling */
    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {{
        color: var(--text-muted) !important;
    }}

    /* 3D Privacy Status Card in Sidebar */
    .privacy-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: var(--card-shadow);
        margin-bottom: 12px;
    }}

    /* 3D Recessed Memory Cards */
    .memory-item {{
        background: var(--memory-bg);
        border: 1px solid var(--memory-border);
        border-radius: 14px;
        padding: 10px 13px;
        margin-bottom: 8px;
        font-size: 13px;
        color: var(--text-primary);
        box-shadow: var(--memory-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .memory-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
    }}
    .memory-label {{
        font-size: 11px;
        font-weight: 700;
        color: var(--memory-label);
        text-transform: uppercase;
        letter-spacing: .7px;
        margin-top: 14px;
        margin-bottom: 8px;
    }}

    /* 3D Tactile Buttons */
    .stButton > button {{
        width: 100%;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        padding: 11px 18px !important;
        background: var(--btn-primary-bg) !important;
        color: var(--btn-primary-text) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: var(--btn-primary-shadow) !important;
        transform: perspective(500px) translateZ(0);
        transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }}
    .stButton > button:hover {{
        box-shadow: var(--btn-primary-hover) !important;
        transform: perspective(500px) translateY(-2px) scale(1.01) !important;
        color: var(--btn-primary-text) !important;
    }}
    .stButton > button:active {{
        transform: perspective(500px) translateY(2px) scale(0.99) !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.25) !important;
    }}

    /* Streamlit Chat Messages 3D Glass Cards */
    [data-testid="stChatMessage"] {{
        border-radius: 18px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
        border: 1px solid var(--ai-msg-border) !important;
        background: var(--ai-msg-bg) !important;
        color: var(--ai-msg-text) !important;
        box-shadow: var(--ai-msg-shadow) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stChatMessage"]:hover {{
        transform: translateY(-1px);
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code {{
        color: var(--ai-msg-text) !important;
    }}

    /* User Message distinct 3D gradient bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: var(--user-msg-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
        box-shadow: var(--user-msg-shadow) !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) code {{
        color: #ffffff !important;
    }}

    /* 3D Chat Input */
    [data-testid="stChatInput"] {{
        border-radius: 18px !important;
        border: 1px solid var(--input-border) !important;
        background: var(--input-bg) !important;
        box-shadow: var(--input-shadow) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), var(--input-shadow) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: var(--input-text) !important;
    }}

    /* Form Inputs and Selectboxes */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {{
        background: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        border-radius: 12px !important;
        color: var(--input-text) !important;
    }}
    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {{
        color: var(--input-text) !important;
    }}
    </style>
    """

st.markdown(get_custom_css(st.session_state.theme_mode), unsafe_allow_html=True)


def get_invoke_messages(quick: bool = True):
    """Core Hazu identity + optional custom system instructions + recent chat history."""
    core_identity = (
        "You are Hazu, a friendly and highly capable AI assistant built by Haziq. "
        "Your name is always Hazu. Whenever anyone asks who you are, what your name is, "
        "who made you, or who built/created you, you must always state clearly and proudly: "
        "'I am Hazu, built by Haziq.' "
        "Never say you were built by Google or OpenAI. You are Hazu, built by Haziq."
    )
    custom_role = (st.session_state.system_role or "").strip()
    system_content = f"{core_identity}\n\nAdditional user instructions:\n{custom_role}" if custom_role else core_identity

    prefix = [SystemMessage(content=system_content)]
    history = list(st.session_state.messages)
    if quick:
        history = history[-8:]
    return prefix + history


def extract_plain_text(response):
    """Return ONLY reply text — strip JSON blocks / signatures."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text
    return str(content)


def clean_display_text(text):
    """Clean stringified response payloads."""
    if not isinstance(text, str):
        return str(text)
    stripped = text.strip()
    if stripped.startswith("[{"):
        try:
            import json
            blocks = json.loads(stripped)
            parts = [
                b.get("text", "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
    return text


def ask_gemini(model_name, temperature, quick: bool = True):
    """Invoke Gemini using the current user's session-only API key."""
    candidates = []
    for m in [model_name, "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"]:
        if m and m not in candidates:
            candidates.append(m)

    last_error = None
    for candidate in candidates:
        try:
            kwargs = dict(
                model=candidate,
                google_api_key=st.session_state.api_key,
                temperature=temperature,
            )
            if quick:
                kwargs["max_output_tokens"] = 800
            chat = ChatGoogleGenerativeAI(**kwargs)
            response = chat.invoke(get_invoke_messages(quick=quick))
            return extract_plain_text(response), candidate
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "NOT_FOUND" in err_str or "404" in err_str or "no longer available" in err_str:
                continue
            break
    raise last_error


def short_error(e):
    err = str(e)
    if "API key" in err or "API_KEY" in err or "401" in err or "403" in err:
        return "Your Gemini API Key looks invalid or expired. Click 'Disconnect / Change Key' in the sidebar and enter a valid key."
    if "NOT_FOUND" in err or "404" in err or "no longer available" in err:
        return "The requested Gemini model is currently unavailable. Please select 'gemini-2.5-flash' in Settings."
    return err[:300]


# ============================================================ SCREEN 1: PRIVATE KEY ENTRANCE
# For public safety: Every visitor starts locked. Keys are NEVER shared or saved to disk.
if not st.session_state.unlocked or not st.session_state.api_key:
    st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

    _, centre, _ = st.columns([1, 2.2, 1])
    with centre:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="key-card">
                <div style="font-size:64px; filter: drop-shadow(0 8px 16px rgba(99,102,241,0.35));">🤖</div>
                <h1>Your Hazu</h1>
                <p>Private & Secure Gemini AI Assistant with 3D Intelligence</p>
                <div style="margin: 12px 0;">
                    <span class="key-badge">✦ 100% Client-Side Private • Bring Your Own Key • Zero Storage</span>
                </div>
                <p style="margin-top:14px; font-size: 14px;">
                    This assistant is publicly accessible. For your complete security and privacy,
                    each user provides their <b>own Gemini API key</b>. Your key is stored
                    <b>only in memory for this session</b> — never written to disk, never shared,
                    and inaccessible to any other user.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔑 Enter your Gemini API Key")
        entered_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your private Gemini API key (starts with AIza…)",
            label_visibility="collapsed",
        )

        if st.button("🚀 Unlock My Assistant", use_container_width=True):
            cleaned_key = (entered_key or "").strip()
            if not cleaned_key:
                st.error("Please paste your Gemini API Key to continue.")
            elif len(cleaned_key) < 20:
                st.error("The key looks too short. Please paste a valid Gemini API Key (starts with AIza...).")
            else:
                # Kept strictly in this user's isolated session in memory
                st.session_state.api_key = cleaned_key
                st.session_state.unlocked = True
                st.success("✅ Connected securely! Opening your assistant...")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🔒 **Privacy Guarantee**: Your key is never logged or saved to the server. You can disconnect at any time.")
        st.caption("Don't have an API key? Get one free at Google AI Studio → [aistudio.google.com](https://aistudio.google.com)")

    st.stop()


# ============================================================ SCREEN 2: CHATBOT (ACTIVE SESSION)
with st.sidebar:
    st.markdown("## 🤖 Your Hazu")

    # 3D Privacy Status Card
    masked = st.session_state.api_key[-4:] if st.session_state.api_key else ""
    st.markdown(
        f"""
        <div class="privacy-card">
            <div style="font-size:11px; font-weight:700; color:var(--memory-label); text-transform:uppercase;">Security & Privacy</div>
            <div style="font-size:15px; font-weight:700; margin:4px 0 4px 0; color:#10b981;">🛡️ Private Session Active</div>
            <div style="font-size:12px; color:var(--text-muted);">Active Key: ••••{masked}</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Not saved to server disk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔒 Disconnect / Change Key", use_container_width=True, help="Wipe key from session memory and lock"):
        st.session_state.api_key = ""
        st.session_state.unlocked = False
        st.session_state.messages = []
        st.session_state.last_error = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎨 Appearance & 3D Theme")
    theme_options = ["Auto (System)", "🌙 Dark Mode", "☀️ Light Mode"]
    selected_theme = st.selectbox(
        "Theme mode",
        options=theme_options,
        index=theme_options.index(st.session_state.theme_mode) if st.session_state.theme_mode in theme_options else 0,
        help="Auto (System) automatically matches your computer's Light or Dark mode. You can also manually switch.",
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"],
        index=0,
        help="gemini-2.5-flash is ultra-fast and reliable for quick replies.",
    )
    quick_mode = st.checkbox(
        "⚡ Quick reply mode",
        value=True,
        help="ON = only recent chat is sent + answers kept short. Turn OFF for long, detailed answers.",
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.5, 0.1)
    system_role = st.text_area(
        "Custom instructions (optional)",
        value=st.session_state.system_role,
        placeholder="Add preferences (e.g. 'Keep answers concise', 'Format as bullet points'). Identity remains 'Hazu, built by Haziq'.",
        height=80,
    )
    st.session_state.system_role = system_role

    st.markdown("---")
    st.markdown("### 🧠 Memory")
    memory = st.session_state.messages
    st.caption(f"{len(memory)} messages in your private session")
    if st.button("🗑️ Clear memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_error = ""
        st.rerun()
    if memory:
        history_text = "\n\n".join(
            f"{'You' if isinstance(m, HumanMessage) else 'Hazu'}: {clean_display_text(m.content)}"
            for m in memory
        )
        st.download_button(
            "⬇️ Download chat history",
            data=history_text,
            file_name=f"hazu-chat-{datetime.now():%Y%m%d-%H%M}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown('<p class="memory-label">Recent conversation</p>', unsafe_allow_html=True)
        for msg in reversed(memory[-10:]):
            role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Hazu"
            preview = clean_display_text(msg.content)
            preview = preview if len(preview) <= 120 else preview[:120] + "…"
            st.markdown(f'<div class="memory-item"><b>{role}:</b><br>{preview}</div>', unsafe_allow_html=True)
    else:
        st.info("No chat memory yet. Start chatting and it will appear here.")

# ------------------------------------------------------------ chatbot header
st.markdown(
    """
    <div class="hazu-header">
        <h1>🤖 Your Hazu</h1>
        <p>✨ <b>I am Hazu, built by Haziq</b> — Ask me anything! Your session is completely private.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

# Welcome suggestions when chat is empty
if not st.session_state.messages:
    st.markdown("#### 👋 Try one of these:")
    cols = st.columns(3)
    suggestions = [
        ("🤖 Who are you?", "Who are you and who built you?"),
        ("💡 Explain AI simply", "Explain AI in simple words"),
        ("📚 Study tips", "Give me 5 effective study tips"),
    ]
    for col, (label, full_prompt) in zip(cols, suggestions):
        with col:
            if st.button(label, key=label, use_container_width=True):
                st.session_state.pending_prompt = full_prompt
                st.rerun()

# ------------------------------------------------------------ render history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(clean_display_text(msg.content))

# --------------------------------------------------------------- chat input
typed = st.chat_input("Ask Hazu anything…")
prompt = typed if typed else st.session_state.pending_prompt
if prompt:
    st.session_state.pending_prompt = ""
    st.session_state.last_error = ""
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, _used_model = ask_gemini(model_name, temperature, quick=quick_mode)
                st.write(answer)
                st.session_state.messages.append(AIMessage(content=answer))
            except Exception as e:
                st.session_state.last_error = f"⚠️ I couldn't reach Gemini. {short_error(e)}"
                st.error(st.session_state.last_error)
