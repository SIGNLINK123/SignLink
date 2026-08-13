import cv2
import os
import time

# --- CONFIGURACIÓN DE GRABACIÓN ---
MODO_CAPTURA = "LETRA"  # Opciones: "PALABRA" o "LETRA"
ETIQUETA = "Y"          # Nombre de la palabra o la letra que van a grabar
CANTIDAD_VIDEOS = 30    # Cuántos videos quieren grabar por tanda
FRAMES_POR_VIDEO = 30   # Duración del gesto (~1 a 1.5 segundos)

# Pedir etiqueta y modo al usuario en tiempo de ejecución
modo_input = input("Elige modo (PALABRA/LETRA) [LETRA]: ").strip().upper()
if modo_input in ("PALABRA", "LETRA"):
    MODO_CAPTURA = modo_input

etiqueta_input = input("Escribe la etiqueta que quieres grabar: ").strip()
if etiqueta_input:
    ETIQUETA = etiqueta_input

# Normalizar nombre: si es modo LETRA, forzar a mayúscula limpia (ej: 'a' -> 'A')
NOMBRE_CARPETA = ETIQUETA.upper() if MODO_CAPTURA == "LETRA" else ETIQUETA.lower()

# Estructura de destino
CARPETA_DESTINO = f"videos_dataset/{NOMBRE_CARPETA}"
os.makedirs(CARPETA_DESTINO, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara. Asegúrate de que esté conectada y libre.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)

print(f"🎬 MODO SELECCIONADO: {MODO_CAPTURA}")
print(f"📁 Grabando etiqueta: '{NOMBRE_CARPETA}' en '{CARPETA_DESTINO}'")
print("Presiona la tecla 'S' (Start) para iniciar la sesión de grabación...")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    cv2.putText(frame, f"MODO: {MODO_CAPTURA} | ETIQUETA: {NOMBRE_CARPETA}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "PRESIONA 'S' PARA EMPEZAR", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    cv2.imshow("SignLink V2 - Captura de Video", frame)
    if cv2.waitKey(1) & 0xFF in (ord('s'), ord('S')):
        break

# Ciclo automático de grabación
for num_video in range(1, CANTIDAD_VIDEOS + 1):
    # Cuenta regresiva de preparación
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Preparate para {MODO_CAPTURA} '{NOMBRE_CARPETA}' (#{num_video}) en... {i}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("SignLink V2 - Captura de Video", frame)
        cv2.waitKey(700)
        
    print(f"🎥 Grabando {MODO_CAPTURA.lower()} {num_video}/{CANTIDAD_VIDEOS}...")
    
    cuadros_video = []
    
    # Capturar la ráfaga de frames
    for frame_idx in range(FRAMES_POR_VIDEO):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Error al capturar el frame {frame_idx + 1} del video {num_video}.")
            break
        frame = cv2.flip(frame, 1)
        
        # Guardar copia limpia para la base de datos
        cuadros_video.append(frame.copy())
        
        # Indicador de grabado
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame, f"GRABANDO {MODO_CAPTURA}: {NOMBRE_CARPETA}", (55, 37), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("SignLink V2 - Captura de Video", frame)
        cv2.waitKey(30)
        
    if len(cuadros_video) == 0:
        print(f"❌ No se grabaron cuadros para {NOMBRE_CARPETA}_{num_video}. Se omite este video.")
        continue
    
    # Guardar video .avi
    ruta_video = os.path.join(CARPETA_DESTINO, f"{NOMBRE_CARPETA}_{num_video}.avi")
    alto, ancho, _ = cuadros_video[0].shape
    out = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (ancho, alto))
    
    for f in cuadros_video:
        out.write(f)
    out.release()
    
    print(f"💾 Guardado: {ruta_video}")

print(f"\n🎉 ¡Grabación de '{NOMBRE_CARPETA}' completada con éxito!")
cap.release()
cv2.destroyAllWindows()