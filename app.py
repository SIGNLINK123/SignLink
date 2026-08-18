import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import av

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SignLink - Traductor de Lengua de Señas",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS AVANZADO Y ESTILIZADO (GLASSMORPHISM & MODO OSCURO) ---
st.markdown("""
    <style>
    /* Fondo principal y reset */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Panel lateral personalizado */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }

    /* Encabezado Principal */
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    /* Tarjetas tipo Glassmorphism */
    .glass-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Caja de Traducción Principal */
    .translation-box {
        background: #0f172a;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 18px;
        min-height: 80px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #38bdf8;
        word-break: break-word;
    }

    /* Badges e Indicadores */
    .badge-info {
        background-color: #0284c7;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Botones estilizados */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    /* Ocultar elementos nativos innecesarios de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTES ---
CARPETA_DATASET = "videos_dataset"
ARCHIVO_CACHE = "dataset_secuencias_cache.npy"
DIMENSION_COORDENADAS = 63
FRAMES_POR_VIDEO = 30
LETRAS_ABC = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# --- CARGAR CACHÉ DEL DATASET ---
@st.cache_resource
def cargar_base_datos():
    if os.path.exists(ARCHIVO_CACHE):
        return np.load(ARCHIVO_CACHE, allow_pickle=True).item()
    return {}

base_datos_videos = cargar_base_datos()

# --- FUNCIONES MATEMÁTICAS / PROCESAMIENTO ---
def normalizar_landmarks(hand_landmarks):
    m_x = hand_landmarks.landmark[0].x
    m_y = hand_landmarks.landmark[0].y
    m_z = hand_landmarks.landmark[0].z

    ref_x = hand_landmarks.landmark[9].x - m_x
    ref_y = hand_landmarks.landmark[9].y - m_y
    ref_z = hand_landmarks.landmark[9].z - m_z
    escala = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    if escala == 0:
        escala = 1.0

    puntos = []
    for lm in hand_landmarks.landmark:
        puntos.extend([
            (lm.x - m_x) / escala,
            (lm.y - m_y) / escala,
            (lm.z - m_z) / escala
        ])
    return np.array(puntos)

def calcular_similitud_coseno(vec1, vec2):
    v1_flat = vec1.flatten()
    v2_flat = vec2.flatten()
    norm1 = np.linalg.norm(v1_flat)
    norm2 = np.linalg.norm(v2_flat)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1_flat, v2_flat) / (norm1 * norm2)

def forma_promedio(secuencia):
    n = len(secuencia)
    borde = max(1, n // 5)
    recorte = secuencia[borde:n - borde] if n - 2 * borde > 0 else secuencia
    frames_utiles = [f for f in recorte if np.count_nonzero(f) > 0]
    if not frames_utiles:
        frames_utiles = [f for f in secuencia if np.count_nonzero(f) > 0]
    if not frames_utiles:
        return np.zeros(DIMENSION_COORDENADAS)
    return np.mean(frames_utiles, axis=0)

def calcular_similitud_forma(secuencia_actual, secuencia_referencia):
    return calcular_similitud_coseno(forma_promedio(secuencia_actual), forma_promedio(secuencia_referencia))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- EN LA SECCIÓN DEL PROCESADOR (SignLanguageProcessor) ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self, base_datos):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5, # Reducir ligeramente para ganar velocidad
            min_tracking_confidence=0.5
        )
        self.base_datos = base_datos
        self.buffer_fotogramas = []
        self.frase_traducida = []
        self.ultima_palabra = ""
        self.tiempo_bloqueo = 0.0
        self.modo_actual = "LETRAS"
        self.ultimo_texto_estado = "Esperando seña..."
        self.frame_counter = 0  # Contador para saltar procesamiento pesado

    # ... (mantén tus métodos agregar_espacio, borrar_ultimo, etc.) ...

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        self.frame_counter += 1

        # Ejecutar MediaPipe solo cada 2 fotogramas para ganar fluidez
        if self.frame_counter % 2 == 0:
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            coordenadas_frame = np.zeros(DIMENSION_COORDENADAS)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coordenadas_frame = normalizar_landmarks(hand_landmarks)

            self.buffer_fotogramas.append(coordenadas_frame)
            if len(self.buffer_fotogramas) > FRAMES_POR_VIDEO:
                self.buffer_fotogramas.pop(0)

            tiempo_espera = 0.8 if self.modo_actual == "LETRAS" else 1.2
            if (len(self.buffer_fotogramas) == FRAMES_POR_VIDEO
                    and (time.time() - self.tiempo_bloqueo > tiempo_espera)
                    and self.base_datos):

                rafaga_actual = np.array(self.buffer_fotogramas)
                mejor_palabra = "Desconocido"
                max_similitud = -1.0
                segunda_similitud = -1.0

                if np.count_nonzero(rafaga_actual) > (FRAMES_POR_VIDEO * DIMENSION_COORDENADAS * 0.5):
                    for palabra, videos_referencia in self.base_datos.items():
                        es_letra = palabra.upper() in LETRAS_ABC
                        if self.modo_actual == "LETRAS" and not es_letra:
                            continue
                        elif self.modo_actual == "PALABRAS" and es_letra:
                            continue

                        mejor_de_esta_palabra = -1.0
                        for v_ref in videos_referencia:
                            if self.modo_actual == "LETRAS":
                                similitud = calcular_similitud_forma(rafaga_actual, v_ref)
                            else:
                                similitud = calcular_similitud_coseno(rafaga_actual, v_ref)
                            if similitud > mejor_de_esta_palabra:
                                mejor_de_esta_palabra = similitud

                        if mejor_de_esta_palabra > max_similitud:
                            segunda_similitud = max_similitud
                            max_similitud = mejor_de_esta_palabra
                            mejor_palabra = palabra
                        elif mejor_de_esta_palabra > segunda_similitud:
                            segunda_similitud = mejor_de_esta_palabra

                gana_con_margen = (max_similitud - segunda_similitud) > 0.02
                if max_similitud > 0.90 and mejor_palabra != "Desconocido" and gana_con_margen:
                    self.ultimo_texto_estado = f"DETECTADO: {mejor_palabra.upper()} ({int(max_similitud * 100)}%)"
                    if mejor_palabra != self.ultima_palabra:
                        self.frase_traducida.append(mejor_palabra)
                        self.ultima_palabra = mejor_palabra
                        self.tiempo_bloqueo = time.time()
                else:
                    self.ultimo_texto_estado = "Buscando seña..."
                    self.ultima_palabra = ""

        # Overlays en pantalla (fuera del 'if' para dibujarse en cada frame)
        cv2.rectangle(image, (0, 0), (image.shape[1], 45), (15, 23, 42), -1)
        cv2.putText(image, self.ultimo_texto_estado, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        texto_frase = self.texto_actual()
        alto = image.shape[0]
        cv2.rectangle(image, (0, alto - 45), (image.shape[1], alto), (255, 255, 255), -1)
        cv2.putText(image, f"Traduccion: {texto_frase}", (15, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --- EN LA SECCIÓN DE WEBRTC EN STREAMLIT ---
ctx = webrtc_streamer(
    key="sign-language-v2",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: SignLanguageProcessor(base_datos_videos),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "max": 1920},
            "height": {"ideal": 720, "max": 1080},
            "frameRate": {"ideal": 30, "max": 60}
        },
        "audio": False
    },
    async_processing=True  # Asincrónico para no bloquear el hilo de renderizado
)

        # Overlays estilizados sobre el video
        cv2.rectangle(image, (0, 0), (image.shape[1], 45), (15, 23, 42), -1)
        cv2.putText(image, self.ultimo_texto_estado, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_ia, 2)
        
        texto_frase = self.texto_actual()
        alto = image.shape[0]
        cv2.rectangle(image, (0, alto - 45), (image.shape[1], alto), (255, 255, 255), -1)
        cv2.putText(image, f"Traduccion: {texto_frase}", (15, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- SIDEBAR (INFORMACIÓN DEL PROYECTO & CONFIGURACIÓN) ---
with st.sidebar:
    st.markdown("## 🤟 SignLink V2")
    st.caption("Inteligencia Artificial para Escuelas Inclusivas")
    st.markdown("---")
    
    st.markdown("### 📌 Sobre el Proyecto")
    st.write(
        "Herramienta interactiva diseñada para romper las barreras de comunicación "
        "entre estudiantes sordos, compañeros y profesores mediante visión por computadora."
    )
    
    with st.expander("👥 Equipo Desarrollador"):
        st.write("""
        * Susana Fontecha
        * Miguel Gaviria
        * Isabela Meneses
        * Sh'muel Ospina
        * Miguel Rivera
        """)
        
    with st.expander("⚙️ ¿Cómo funciona?"):
        st.markdown("""
        1. **Captura:** MediaPipe extrae los puntos de la mano.
        2. **Normalización:** Se ajusta la escala y centro.
        3. **Clasificación:** Comparación vectorial mediante similitud coseno.
        """)

    st.markdown("---")
    st.markdown("📞 **Soporte:** `305 3805159` | `302 6980645`")

# --- CONTENIDO PRINCIPAL ---
st.markdown("""
    <div class="main-header">
        <h1>✋ SignLink - Traductor de Lengua de Señas</h1>
        <p>Plataforma en tiempo real para la inclusión y accesibilidad en el aula.</p>
    </div>
""", unsafe_allow_html=True)

if not base_datos_videos:
    st.error(f"⚠️ No se encontró el archivo de caché `{ARCHIVO_CACHE}`. Carga la base de datos para comenzar.")

col_video, col_panel = st.columns([1.6, 1], gap="large")

with col_video:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📹 Entrada de Cámara Web")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="sign-language-v2",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: SignLanguageProcessor(base_datos_videos),
        media_stream_constraints={"video": True, "audio": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_panel:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Configuración de Modo")
    
    modo = st.radio("Selecciona la modalidad:", ["LETRAS", "PALABRAS"], horizontal=True)
    if ctx.video_processor:
        ctx.video_processor.modo_actual = modo
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📝 Traducción en Tiempo Real")
    
    cuadro_texto = st.empty()
    if ctx.video_processor:
        cuadro_texto.markdown(f'<div class="translation-box">{ctx.video_processor.texto_actual()}</div>', unsafe_allow_html=True)
    else:
        cuadro_texto.markdown('<div class="translation-box" style="color: #64748b;">(Inicia la cámara para comenzar)</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Controles de texto
    b1, b2, b3 = st.columns(3)
    if b1.button("␣ Espacio", use_container_width=True) and ctx.video_processor:
        ctx.video_processor.agregar_espacio()
    if b2.button("⌫ Borrar", use_container_width=True) and ctx.video_processor:
        ctx.video_processor.borrar_ultimo()
    if b3.button("🗑️ Limpiar", use_container_width=True, type="primary") and ctx.video_processor:
        ctx.video_processor.limpiar_todo()

    st.markdown('</div>', unsafe_allow_html=True)
