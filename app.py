import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SignLink V2 - AI Sign Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO (ESTILO MODERNO / MODO OSCURO) ---
st.markdown("""
    <style>
    /* Fondo principal y tipografía */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Panel lateral */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Contenedores y Tarjetas */
    .css-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    
    /* Banner de Seña Detectada */
    .detection-box {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .detection-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
    }
    .detection-word {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    
    /* Caja de Historial */
    .history-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 6px;
        font-size: 1.2rem;
        min-height: 60px;
        word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN Y CONSTANTES ---
FRAMES_POR_VIDEO = 30
DIMENSION_COORDENADAS = 63
CARPETA_DATASET = "videos_dataset"
ARCHIVO_CACHE = "dataset_secuencias_cache.npy"
LETRAS_ABC = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def normalizar_landmarks(hand_landmarks):
    m_x, m_y, m_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    ref_x = hand_landmarks.landmark[9].x - m_x
    ref_y = hand_landmarks.landmark[9].y - m_y
    ref_z = hand_landmarks.landmark[9].z - m_z
    escala = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    if escala == 0: escala = 1.0
    
    puntos = []
    for lm in hand_landmarks.landmark:
        puntos.extend([(lm.x - m_x) / escala, (lm.y - m_y) / escala, (lm.z - m_z) / escala])
    return np.array(puntos)

def calcular_similitud_coseno(vec1, vec2):
    v1_flat, v2_flat = vec1.flatten(), vec2.flatten()
    norm1, norm2 = np.linalg.norm(v1_flat), np.linalg.norm(v2_flat)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(v1_flat, v2_flat) / (norm1 * norm2))

@st.cache_data
def cargar_dataset():
    if os.path.exists(ARCHIVO_CACHE):
        return np.load(ARCHIVO_CACHE, allow_pickle=True).item()
    return {}

base_datos_videos = cargar_dataset()

# --- ESTADO GLOBAL (SESSION STATE) ---
if "historial" not in st.session_state:
    st.session_state.historial = []
if "ultima_detectada" not in st.session_state:
    st.session_state.ultima_detectada = ""

# --- SIDEBAR / CONTROLES ---
with st.sidebar:
    st.title("🤟 SignLink V2")
    st.caption("Sistema Inteligente de Reconocimiento de Lengua de Señas")
    st.markdown("---")
    
    modo = st.radio("📌 Modo de Operación", ["PALABRAS", "LETRAS"], index=0)
    
    st.markdown("---")
    st.subheader("⚙️ Ajustes de Cámara")
    confianza_min = st.slider("Umbral de Confianza", 0.70, 0.95, 0.86, 0.01)
    
    st.markdown("---")
    if st.button("🗑️ Borrar Última Seña", use_container_width=True):
        if st.session_state.historial:
            st.session_state.historial.pop()
            st.session_state.ultima_detectada = ""
            st.rerun()

    if st.button("❌ Limpiar Todo", use_container_width=True, type="primary"):
        st.session_state.historial = []
        st.session_state.ultima_detectada = ""
        st.rerun()

# --- PANEL PRINCIPAL ---
col_video, col_panel = st.columns([2, 1], gap="medium")

with col_panel:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📊 Estado en Tiempo Real")
    
    # Placeholder para actualización dinámica sin refrescar toda la página
    detector_placeholder = st.empty()
    confianza_bar = st.progress(0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📝 Traducción Acumulada")
    
    # Formateo dinámico según modo (Letras = pegado, Palabras = con espacio)
    texto_traduccion = "".join(st.session_state.historial) if modo == "LETRAS" else " ".join(st.session_state.historial)
    st.markdown(f'<div class="history-box">{texto_traduccion if texto_traduccion else "<i>Esperando señas...</i>"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_video:
    st.markdown("### 🎥 Entrada de Video")
    run_cam = st.checkbox("Activar Cámara", value=True)
    FRAME_WINDOW = st.image([])

# --- PROCESAMIENTO EN VIVO ---
if run_cam:
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
    
    buffer_fotogramas = []
    tiempo_bloqueo = 0.0

    while run_cam:
        success, frame = cap.read()
        if not success:
            st.error("No se pudo acceder a la cámara web.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        coordenadas_frame = np.zeros(DIMENSION_COORDENADAS)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coordenadas_frame = normalizar_landmarks(hand_landmarks)

        buffer_fotogramas.append(coordenadas_frame)
        if len(buffer_fotogramas) > FRAMES_POR_VIDEO:
            buffer_fotogramas.pop(0)

        palabra_detectada = "Detectando..."
        similitud_max = 0.0

        if len(buffer_fotogramas) == FRAMES_POR_VIDEO and (time.time() - tiempo_bloqueo > 1.2):
            rafaga_actual = np.array(buffer_fotogramas)
            
            if np.count_nonzero(rafaga_actual) > (FRAMES_POR_VIDEO * DIMENSION_COORDENADAS * 0.5):
                for palabra, videos_ref in base_datos_videos.items():
                    es_letra = palabra.upper() in LETRAS_ABC
                    if modo == "LETRAS" and not es_letra: continue
                    elif modo == "PALABRAS" and es_letra: continue

                    for v_ref in videos_ref:
                        sim = calcular_similitud_coseno(rafaga_actual, v_ref)
                        if sim > similitud_max:
                            similitud_max = sim
                            palabra_detectada = palabra

            if similitud_max > confianza_min and palabra_detectada != "Detectando...":
                if palabra_detectada != st.session_state.ultima_detectada:
                    st.session_state.historial.append(palabra_detectada)
                    st.session_state.ultima_detectada = palabra_detectada
                    tiempo_bloqueo = time.time()
                    st.rerun()

        # Renderizar datos en el panel lateral
        confianza_pct = int(similitud_max * 100) if similitud_max > 0 else 0
        confianza_bar.progress(min(confianza_pct, 100))
        
        texto_tarjeta = palabra_detectada.upper() if similitud_max > confianza_min else "ESCANEAR..."
        detector_placeholder.markdown(f"""
            <div class="detection-box">
                <div class="detection-title">SEÑA RECONOCIDA</div>
                <div class="detection-word">{texto_tarjeta}</div>
                <small>Confianza: {confianza_pct}%</small>
            </div>
        """, unsafe_allow_html=True)

        # Mostrar video en vivo en la interfaz
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    hands.close()
