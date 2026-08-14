import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import av

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Traductor de Lenguaje de Señas - Escuelas Inclusivas", layout="wide")

st.title("✋ Traductor de Lenguaje de Señas en Tiempo Real")
st.markdown("### Proyecto escolar para la inclusión de estudiantes sordos en las aulas.")

with st.container(border=True):
    st.markdown("## 📌 ¿Para qué sirve este proyecto?")
    st.write(
        "Este traductor usa una cámara y una inteligencia artificial entrenada en Python "
        "para reconocer lengua de señas y convertirla en texto al instante. La idea es simple: "
        "que una conversación en clase no se detenga porque uno de los estudiantes no puede "
        "escuchar o hablar de la forma tradicional."
    )

    st.markdown("#### 👥 ¿A quiénes ayudamos?")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown("**🧑‍🎓 Estudiantes sordos**\n\nPara que puedan comunicarse con sus compañeros sin depender siempre de un intérprete.")
    col_b.markdown("**🧑‍🏫 Profesores**\n\nPara entender lo que un estudiante les está diciendo en lengua de señas, en el momento.")
    col_c.markdown("**🧑‍🤝‍🧑 Compañeros de clase**\n\nPara que todo el salón pueda participar en la misma conversación, sin barreras.")

    st.markdown("#### 💡 ¿Por qué es importante?")
    st.write(
        "Muchos estudiantes sordos no cuentan con un intérprete de lengua de señas disponible "
        "todo el tiempo en su escuela. Eso los deja por fuera de conversaciones cotidianas, "
        "explicaciones espontáneas o simplemente hacer una pregunta rápida en clase. Este proyecto "
        "busca reducir esa barrera usando solo una cámara común, sin equipos costosos."
    )

    st.markdown("#### ⚙️ ¿Cómo funciona?")
    paso1, paso2, paso3 = st.columns(3)
    with paso1:
        st.markdown("**1. Captura**")
        st.caption("La cámara detecta la mano y sus puntos clave mientras se hace la seña.")
    with paso2:
        st.markdown("**2. Reconocimiento**")
        st.caption("La IA compara ese movimiento contra las señas que aprendió previamente.")
    with paso3:
        st.markdown("**3. Traducción**")
        st.caption("El texto aparece al instante, tanto sobre el video como en el panel de resultado.")

    st.markdown("#### 👨‍💻 Quiénes hicimos este proyecto")
    st.write("Susana Fontecha · Miguel Gaviria · Isabela Meneses · Sh'muel Ospina · Miguel Rivera")

    st.markdown("#### 📞 Soporte")
    st.write("¿Dudas o problemas usando el traductor? Escríbenos al 305 3805159 o 302 6980645")

st.write("")

# --- CONSTANTES (igual que en video_en_vivo.py) ---
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

if not base_datos_videos:
    st.error(
        f"⚠️ No se encontró '{ARCHIVO_CACHE}' junto a app.py. "
        "El traductor necesita este archivo para reconocer señas."
    )

# --- FUNCIONES DE PROCESAMIENTO (tomadas de video_en_vivo.py) ---
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
    """Promedia los cuadros centrales de una secuencia (ignora el 20% inicial y final,
    donde la mano suele estar entrando o saliendo del gesto), y descarta cuadros
    donde no se detectó ninguna mano. Esto vuelve la comparación tolerante a que
    el gesto no esté perfectamente alineado en el tiempo."""
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
    """Compara la 'forma promedio' de dos secuencias. Ideal para letras (posturas
    fijas de la mano), donde importa la forma y no tanto el momento exacto."""
    return calcular_similitud_coseno(forma_promedio(secuencia_actual), forma_promedio(secuencia_referencia))


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self, base_datos):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6
        )
        self.base_datos = base_datos
        self.buffer_fotogramas = []
        self.frase_traducida = []
        self.ultima_palabra = ""
        self.tiempo_bloqueo = 0.0
        self.modo_actual = "LETRAS"  # se actualiza desde la barra lateral
        self.ultimo_texto_estado = "Esperando seña..."

    # --- controles llamados desde los botones de Streamlit ---
    def agregar_espacio(self):
        self.frase_traducida.append(" ")
        self.ultima_palabra = ""

    def borrar_ultimo(self):
        if self.frase_traducida:
            self.frase_traducida.pop()
        self.ultima_palabra = ""

    def limpiar_todo(self):
        self.frase_traducida = []
        self.buffer_fotogramas = []
        self.ultima_palabra = ""

    def texto_actual(self):
        if self.modo_actual == "LETRAS":
            return "".join(self.frase_traducida) if self.frase_traducida else "(esperando deletreo...)"
        return " ".join(self.frase_traducida) if self.frase_traducida else "(esperando gesto...)"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
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

        texto_ia = "Escaneando gesto..."
        color_ia = (0, 255, 255)
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

            # Umbral más exigente + exigir que gane con margen claro sobre la 2da opción,
            # para no confundir señas parecidas entre sí.
            gana_con_margen = (max_similitud - segunda_similitud) > 0.02
            if max_similitud > 0.90 and mejor_palabra != "Desconocido" and gana_con_margen:
                texto_ia = f"DETECTADO: {mejor_palabra.upper()} ({int(max_similitud * 100)}%)"
                color_ia = (0, 255, 0)

                if mejor_palabra != self.ultima_palabra:
                    self.frase_traducida.append(mejor_palabra)
                    self.ultima_palabra = mejor_palabra
                    self.tiempo_bloqueo = time.time()
            else:
                texto_ia = "Buscando seña..."
                color_ia = (0, 0, 255)
                self.ultima_palabra = ""

            self.ultimo_texto_estado = texto_ia

        # --- overlays en el video, igual que la versión de escritorio ---
        cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(image, self.ultimo_texto_estado, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ia, 2)
        cv2.putText(
            image, f"MODO: {self.modo_actual}", (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1
        )

        texto_frase = self.texto_actual()
        alto = image.shape[0]
        cv2.rectangle(image, (0, alto - 50), (image.shape[1], alto), (255, 255, 255), -1)
        cv2.putText(image, f"Resultado: {texto_frase}", (10, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --- INTERFAZ VISUAL DE STREAMLIT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Cámara en Directo")
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="sign-language",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: SignLanguageProcessor(base_datos_videos),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📖 Traducción Actual")

    modo = st.radio("Modo de reconocimiento", ["LETRAS", "PALABRAS"], horizontal=True)
    if ctx.video_processor:
        ctx.video_processor.modo_actual = modo

    st.info("Colócate frente a la cámara y realiza una seña registrada en el dataset.")

    cuadro_texto = st.empty()

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("␣ Espacio") and ctx.video_processor:
        ctx.video_processor.agregar_espacio()
    if b2.button("⌫ Borrar") and ctx.video_processor:
        ctx.video_processor.borrar_ultimo()
    if b3.button("🗑️ Limpiar") and ctx.video_processor:
        ctx.video_processor.limpiar_todo()
    if b4.button("🔄 Actualizar texto"):
        pass  # solo fuerza el rerender de Streamlit

    if ctx.video_processor:
        cuadro_texto.markdown(f"### **Texto:** *{ctx.video_processor.texto_actual()}*")
    else:
        cuadro_texto.markdown("### **Texto:** *(Inicia la cámara para comenzar)*")

    st.caption("El texto se actualiza directamente sobre el video. Usa '🔄 Actualizar texto' para reflejarlo también aquí.")

    st.markdown("---")
    st.markdown("### 💡 Enfoque Pedagógico")
    st.write("Este sistema apoya la integración escolar permitiendo una comunicación visual accesible e inmediata.")
