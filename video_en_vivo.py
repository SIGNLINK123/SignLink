import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURACIÓN DE RÁFAGA Y MODO ---
FRAMES_POR_VIDEO = 30
DIMENSION_COORDENADAS = 63
CARPETA_DATASET = "videos_dataset"
ARCHIVO_CACHE = "dataset_secuencias_cache.npy"

MODO_ACTUAL = "LETRAS"  # Alterna entre "PALABRAS" y "LETRAS"
LETRAS_ABC = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def normalizar_landmarks(hand_landmarks):
    m_x = hand_landmarks.landmark[0].x
    m_y = hand_landmarks.landmark[0].y
    m_z = hand_landmarks.landmark[0].z
    
    ref_x = hand_landmarks.landmark[9].x - m_x
    ref_y = hand_landmarks.landmark[9].y - m_y
    ref_z = hand_landmarks.landmark[9].z - m_z
    escala = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    if escala == 0: escala = 1.0
    
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
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1_flat, v2_flat) / (norm1 * norm2)

base_datos_videos = {} 

# --- SISTEMA DE CACHÉ ---
if os.path.exists(ARCHIVO_CACHE):
    print("⚡ Cargando base de datos desde la caché...")
    base_datos_videos = np.load(ARCHIVO_CACHE, allow_pickle=True).item()
    print("✅ Base de datos cargada.")
else:
    print("📦 Procesando dataset...")
    if os.path.exists(CARPETA_DATASET):
        mp_hands_init = mp.solutions.hands
        hands_init = mp_hands_init.Hands(static_image_mode=True, max_num_hands=1)
        
        for palabra in os.listdir(CARPETA_DATASET):
            ruta_palabra = os.path.join(CARPETA_DATASET, palabra)
            if not os.path.isdir(ruta_palabra): continue
            
            base_datos_videos[palabra] = []
            print(f"   Procesando: {palabra}")
            
            for archivo_video in os.listdir(ruta_palabra):
                if not archivo_video.endswith(".avi"): continue
                cap_v = cv2.VideoCapture(os.path.join(ruta_palabra, archivo_video))
                secuencia = []
                
                while cap_v.isOpened():
                    ret, frame = cap_v.read()
                    if not ret: break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands_init.process(rgb)
                    pts = np.zeros(DIMENSION_COORDENADAS)
                    
                    if res.multi_hand_landmarks:
                        pts = normalizar_landmarks(res.multi_hand_landmarks[0])
                        
                    secuencia.append(pts)
                cap_v.release()
                
                if len(secuencia) > FRAMES_POR_VIDEO: secuencia = secuencia[:FRAMES_POR_VIDEO]
                while len(secuencia) < FRAMES_POR_VIDEO: secuencia.append(np.zeros(DIMENSION_COORDENADAS))
                
                base_datos_videos[palabra].append(np.array(secuencia))
        hands_init.close()
        np.save(ARCHIVO_CACHE, base_datos_videos)
    else:
        print(f"❌ No existe '{CARPETA_DATASET}'.")
        exit()

# --- MEDIAPIPE Y CÁMARA ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)

buffer_fotogramas = []
frase_traducida = []
ultima_palabra = ""
tiempo_bloqueo = 0.0

# Variables para la demostración de la seña detectada
cap_demostracion = None
frame_demo = None

print("\n🚀 ¡SignLink V2 EN VIVO (Con Reproducción de Seña)! Ready...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
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
        
    texto_ia = "Escaneando gesto..."
    color_ia = (0, 255, 255)
    tiempo_espera = 0.8 if MODO_ACTUAL == "LETRAS" else 1.2
    
    if len(buffer_fotogramas) == FRAMES_POR_VIDEO and (time.time() - tiempo_bloqueo > tiempo_espera):
        rafaga_actual = np.array(buffer_fotogramas)
        mejor_palabra = "Desconocido"
        max_similitud = -1.0
        
        if np.count_nonzero(rafaga_actual) > (FRAMES_POR_VIDEO * DIMENSION_COORDENADAS * 0.5):
            for palabra, videos_referencia in base_datos_videos.items():
                es_letra = palabra.upper() in LETRAS_ABC
                
                if MODO_ACTUAL == "LETRAS" and not es_letra: continue
                elif MODO_ACTUAL == "PALABRAS" and es_letra: continue
                
                for v_ref in videos_referencia:
                    similitud = calcular_similitud_coseno(rafaga_actual, v_ref)
                    if similitud > max_similitud:
                        max_similitud = similitud
                        mejor_palabra = palabra
        
        if max_similitud > 0.86 and mejor_palabra != "Desconocido":
            texto_ia = f"DETECTADO: {mejor_palabra.upper()} ({int(max_similitud*100)}%)"
            color_ia = (0, 255, 0)
            
            if mejor_palabra != ultima_palabra:
                frase_traducida.append(mejor_palabra)
                ultima_palabra = mejor_palabra
                tiempo_bloqueo = time.time()
                
                # --- CARGAR VIDEO DE DEMOSTRACIÓN DEL DATASET ---
                carpeta_demo = os.path.join(CARPETA_DATASET, mejor_palabra)
                if os.path.exists(carpeta_demo):
                    archivos_demo = [f for f in os.listdir(carpeta_demo) if f.endswith('.avi')]
                    if archivos_demo:
                        ruta_demo = os.path.join(carpeta_demo, archivos_demo[0])
                        if cap_demostracion is not None: cap_demostracion.release()
                        cap_demostracion = cv2.VideoCapture(ruta_demo)
        else:
            texto_ia = "Buscando seña..."
            color_ia = (0, 0, 255)
            ultima_palabra = ""

    # --- REPRODUCIR CUADRO DE DEMOSTRACIÓN (SI EXISTE) ---
    if cap_demostracion is not None and cap_demostracion.isOpened():
        ret_demo, frame_demo = cap_demostracion.read()
        if not ret_demo:
            # Si termina el video de demostración, se reinicia
            cap_demostracion.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_demo, frame_demo = cap_demostracion.read()
        
        if ret_demo and frame_demo is not None:
            # Redimensionar el video de muestra e incrustarlo en la esquina superior derecha
            frame_demo_mini = cv2.resize(frame_demo, (160, 120))
            alto, ancho, _ = frame.shape
            
            # Superponer en la cámara principal
            frame[50:170, ancho-170:ancho-10] = frame_demo_mini
            cv2.rectangle(frame, (ancho-170, 50), (ancho-10, 170), (0, 255, 0), 2)
            cv2.putText(frame, "MUESTRA", (ancho-160, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # --- UI PRINCIPAL ---
    cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
    cv2.putText(frame, texto_ia, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ia, 2)
    
    color_modo = (0, 255, 0) if MODO_ACTUAL == "PALABRAS" else (255, 165, 0)
    cv2.putText(frame, f"MODO: {MODO_ACTUAL} ['M' para cambiar]", (10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_modo, 1)

    if MODO_ACTUAL == "LETRAS":
        texto_frase = "".join(frase_traducida) if frase_traducida else "Esperando deletreo..."
    else:
        texto_frase = " ".join(frase_traducida) if frase_traducida else "Esperando gesto..."
        
    cv2.rectangle(frame, (0, 430), (640, 480), (255, 255, 255), -1)
    cv2.putText(frame, f"Resultado: {texto_frase}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imshow('SignLink V2 - Reconocedor de Video', frame)
    
    # --- TECLAS ---
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'): 
        break
    elif tecla in (ord('m'), ord('M')):
        MODO_ACTUAL = "LETRAS" if MODO_ACTUAL == "PALABRAS" else "PALABRAS"
        print(f"🔄 Modo: {MODO_ACTUAL}")
    elif tecla == 32: # Espacio
        frase_traducida.append(" ")
        ultima_palabra = ""
    elif tecla == 8: # Backspace
        if frase_traducida:
            frase_traducida.pop()
            ultima_palabra = ""
    elif tecla == ord('c'):
        frase_traducida = []
        buffer_fotogramas = []
        ultima_palabra = ""
        if cap_demostracion: cap_demostracion.release()

if cap_demostracion: cap_demostracion.release()
cap.release()
cv2.destroyAllWindows()
hands.close()