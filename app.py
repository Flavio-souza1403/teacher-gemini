import streamlit as st
import google.generativeai as genai
import json
import re
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from io import BytesIO
from tempfile import NamedTemporaryFile
from pydub import AudioSegment

# ========================================================
# CONFIG
# ========================================================
st.set_page_config(page_title="Teacher Gemini v9.1", page_icon="👩‍🏫", layout="wide")

# ========================================================
# CHAVE (Secrets no Cloud)
# ========================================================
API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = st.session_state.get("GEMINI_API_KEY", None)

if not API_KEY or "COLE_SUA_CHAVE" in str(API_KEY):
    st.error("❌ Defina sua chave em st.secrets['GEMINI_API_KEY'] (Streamlit Secrets).")
    st.stop()

# ========================================================
# MODELO (cache)
# ========================================================
@st.cache_resource
def get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"temperature": 0.6, "max_output_tokens": 900},
    )

model = get_model(API_KEY)

# ========================================================
# CENÁRIOS
# ========================================================
CENARIOS = {
    "✈️ Viagem": ["Aeroporto (Imigração)", "Dentro do Avião", "Alfândega"],
    "🏫 Escola": ["Sala de Aula", "Biblioteca", "Secretaria"],
    "🍔 Restaurante": ["Pedir a Comida", "Pagar a Conta"],
    "🏨 Hotel": ["Check-in", "Check-out"],
    "🏙️ Cidade": ["Uber/Táxi", "Perguntar Direção"],
}

# ========================================================
# 1) JSON blindado
# ========================================================
def extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Resposta vazia do modelo")

    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()

    start = t.find("{")
    end = t.rfind("}") + 1
    if start == -1 or end <= 0:
        raise ValueError("Não encontrei bloco JSON na resposta")

    candidate = t[start:end].strip()
    candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
    return json.loads(candidate)

def repair_json_with_model(bad_text: str) -> dict:
    repair_prompt = f"""
Você recebeu um conteúdo que deveria ser JSON, mas está inválido.
Corrija e devolva APENAS um JSON válido com EXATAMENTE estas chaves:
conversation, translation, correction, explanation

Conteúdo inválido:
{bad_text}
"""
    r = model.generate_content(repair_prompt)
    return extract_json(getattr(r, "text", ""))

def perguntar_para_ia(mensagem: str, contexto: str):
    prompt_completo = f"""
Contexto: {contexto}. Você é uma professora de inglês.
Responda ao aluno no personagem do contexto e ajude com o inglês dele.

Responda APENAS no formato JSON abaixo (NÃO use markdown, NÃO use ```):
{{
  "conversation": "Sua resposta em inglês",
  "translation": "Tradução para português",
  "correction": "Frase do aluno corrigida (ou '-' se ok)",
  "explanation": "Dica de gramática em português"
}}

Mensagem do Aluno: "{mensagem}"
"""
    try:
        response = model.generate_content(prompt_completo)
        raw = getattr(response, "text", "")
        try:
            return extract_json(raw)
        except Exception:
            return repair_json_with_model(raw)
    except Exception as e:
        st.error(f"Erro na conexão com o Gemini: {e}")
        return None

# ========================================================
# 2) Áudio (robusto)
# ========================================================
def bytes_to_wav_pcm(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        raise ValueError("audio_bytes vazio")

    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()

def transcrever_audio(audio_bytes: bytes) -> str:
    r = sr.Recognizer()
    try:
        wav_bytes = bytes_to_wav_pcm(audio_bytes)

        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()

            with sr.AudioFile(tmp.name) as source:
                r.adjust_for_ambient_noise(source, duration=0.4)
                audio_data = r.record(source)

            return r.recognize_google(audio_data, language="en-US")
    except Exception as e:
        return f"⚠️ Erro ao transcrever: {e}"

# ========================================================
# 3) Estado
# ========================================================
def init_state():
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
    if "pending_user" not in st.session_state:
        st.session_state.pending_user = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

init_state()

# ========================================================
# SIDEBAR
# ========================================================
with st.sidebar:
    try:
        st.image("teacher2.png", use_container_width=True)
    except Exception:
        st.warning("Imagem teacher2.png não encontrada (ok no deploy se você não enviar).")

    st.title("👩‍🏫 Teacher V9.1 (Gemini Flash)")
    cat = st.selectbox("Escolha o Tema:", list(CENARIOS.keys()))
    cen = st.selectbox("Escolha o Local:", CENARIOS[cat])

    st.divider()
    st.caption("🔐 Use Secrets no Streamlit Cloud para GEMINI_API_KEY.")

    if st.button("🗑️ Reiniciar Tudo"):
        st.session_state.mensagens = []
        st.session_state.pending_user = None
        st.session_state.is_processing = False
        st.rerun()

# ========================================================
# TELA PRINCIPAL
# ========================================================
st.header(f"🇺🇸 Praticando em: {cen}")

if not st.session_state.mensagens:
    st.session_state.mensagens.append({
        "role": "assistant",
        "content": {
            "conversation": "Hello! I am ready to practice. Say something!",
            "translation": "Olá! Estou pronta para praticar. Diga algo!",
            "correction": "-",
            "explanation": "Diga 'Hello' para testar!"
        }
    })

for i, msg in enumerate(st.session_state.mensagens):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            d = msg["content"]
            st.subheader(f"🗣️ {d.get('conversation','')}")
            st.caption(f"🇧🇷 {d.get('translation','')}")

            if st.button("▶️ Ouvir Pronúncia", key=f"play_{i}"):
                tts = gTTS(text=d.get("conversation", ""), lang="en")
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp, format="audio/mp3")

            if d.get("correction", "-") != "-":
                with st.expander("📝 Feedback da Teacher"):
                    st.success(f"✅ {d.get('correction','-')}")
                    st.info(f"💡 {d.get('explanation','')}")

col1, col2 = st.columns([8, 1])
with col1:
    u_input = st.chat_input("Escreva em inglês...")
with col2:
    u_audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key="mic")

if u_audio and not st.session_state.is_processing:
    txt = transcrever_audio(u_audio.get("bytes", b""))
    if txt.startswith("⚠️"):
        st.warning(txt)
    else:
        st.session_state.pending_user = txt

if u_input and not st.session_state.is_processing:
    st.session_state.pending_user = u_input

if st.session_state.pending_user and not st.session_state.is_processing:
    st.session_state.is_processing = True
    user_msg = st.session_state.pending_user
    st.session_state.pending_user = None

    st.session_state.mensagens.append({"role": "user", "content": user_msg})

    with st.spinner("Conectando com o cérebro da Teacher..."):
        resposta = perguntar_para_ia(user_msg, cen)

    if resposta:
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})

    st.session_state.is_processing = False
    st.rerun()
