import cv2
import numpy as np

# Configuración
VIDEO_PATH = "trafico.mp4"
RESIZE_WIDTH = 1000         # Ancho de la imagen. (Resize)
BG_FRAMES = 200             # Cuántos frames usamos para "memorizar" la carretera vacía
THRESH = 30                 # Umbral
AREA_MIN = 600              # Tamaño mínimo del coche para descartar contornos y blobs

# Tracking
MATCH_DIST = 90             # Distancia máxima para decir "es el mismo coche"
MAX_MISSED_FRAMES = 15      # Si no vemos al coche en 15 frames, lo olvidamos

# CLASES Y FUNCIONES
class Coche:
    """Clase para recordar la información de cada vehículo."""
   # next_id = 0
    def __init__(self, id, cx, cy, rect):
        self.id = id                # Identificador
        self.positions = [(cx, cy)] # Historial de dónde ha estado
        self.missed = 0             # Veces que lo hemos perdido de vista
        self.counted = False        # Si ya lo hemos sumado al total
        self.rect = rect            # Caja (x, y, w, h)

    def update(self, cx, cy, rect):
        """Actualiza la posición del coche en el nuevo frame."""
        self.positions.append((cx, cy))
        self.missed = 0             # reiniciamos contador de perdido
        self.rect = rect

    def last_pos(self):
        """Devuelve la última coordenada conocida."""
        return self.positions[-1]

def euclid(p1, p2):
    """Calcula la distancia entre dos puntos (Pitágoras)."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def preprocesar(frame, ancho_objetivo):
    """
    Función auxiliar que redimensiona, pasa a gris y suaviza.
    devuelve la imagen en color redimensionada y la imagen en gris procesada.
    """
    # Resize manteniendo aspect ratio
    h_orig, w_orig = frame.shape[:2]  # Obtenemos alto y ancho originales
    scale = ancho_objetivo / float(w_orig) # Calculamos proporción
    new_h = int(h_orig * scale)
    frame_resized = cv2.resize(frame, (ancho_objetivo, new_h)) # Redimensionamos

    # Pasamos a grises (más fácil de procesar)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) 
    
    # Usamos GaussianBlur para suavizar y quitar ruido de la cámara
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # 5,5: tamaño del kernel
    
    return frame_resized, gray

# APRENDIZAJE DEL FONDO
cap = cv2.VideoCapture(VIDEO_PATH) # Cargamos el video en memoria
bg_acc = None  # Acumulador para calcular el promedio del fondo (matriz)
count = 0      # Contador de frames leídos

# Leemos los 200 primeros frames
while count < BG_FRAMES:
    ret, frame_raw = cap.read() # Leemos un frame (ret -> true si ok)
    if not ret: # si ret == False, no hay frame válido
        break       # Si no hay video, salimos

    # APLICAMOS LA FUNCIÓN DE PREPROCESADO (Resize + Gray + Blur)
    # Esto sustituye a las líneas repetidas de redimensionado y filtrado
    frame, gray = preprocesar(frame_raw, RESIZE_WIDTH)

    # Acumulamos la imagen para hacer el promedio
    if bg_acc is None: # empieza en None para saber si todavía no hemos inicializado el acumulador
        bg_acc = gray.astype(np.float32) # Inicializamos el acumulador en el primer frame
    else:
        bg_acc += gray.astype(np.float32) # Cogemos lo acumulado y sumamos el nuevo frame
    
    count += 1 # cuenta de frames leídos hasta ahora

# Calculamos la media para obtener el fondo "limpio"
if count > 0:
    background = (bg_acc / count).astype(np.uint8) # uint8: (0–255)
else:
    print("Error: No se leyeron frames.")
    exit() # Error si no hay frames

cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rebobinamos al principio

# BUCLE PRINCIPAL (procesamiento frame a frame)
coches_en_escena = [] # Lista de coches activos
next_id = 0           # Identificador para el siguiente coche
final_count = 0       # Cuenta final

while True:
    ret, frame_raw = cap.read() # Leemos frame actual
    if not ret: 
        break               # Fin del video

    # Obtenemos el frame redimensionado y la versión gris suavizada directamente
    frame, gray = preprocesar(frame_raw, RESIZE_WIDTH)
    
    frame_out = frame.copy() # Copia para dibujar
    new_h = frame.shape[0]   # Obtenemos la nueva altura para calcular el área que nos interesa

    # Región de interés
    # Definimos la zona de conteo (del 55% al 85% de la altura)
    y_arriba = int(new_h * 0.55)
    y_abajo = int(new_h * 0.85)
    x_izq = 0
    x_der = RESIZE_WIDTH # Ancho completo

    # SUSTRACCIÓN DE FONDO
    # Diferencia y Umbral
    diff = cv2.absdiff(background, gray) # Diferencia fondo y frame actual
    # Si la diferencia > 30 (umbral), es un objeto (Blanco), si no, es fondo (Negro)
    _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY) #diff(x,y) > THRESH --> 255 (blanco) y diff(x,y) ≤ THRESH --> 0 (negro)

    # Eliminación de ruido
    # MORPH_OPEN: quitamos puntitos pequeños (asfalto) 
    kernel_limpieza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # crea el kernel y genera una elipse porque se adapta mejor a los bordes del coche 
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_limpieza, iterations=1) # limpia las manchas blancas pequeñas
    
    # 2. MORPH_CLOSE ("Pegar"): Unimos huecos dentro del coche
    kernel_relleno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_final = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel_relleno, iterations=2) # kernel_relleno: cuán grandes pueden ser los huecos que se rellenan
    
    # DETECCIÓN DE CONTORNOS
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #lista de arrays, rastrea los bordes de las manchas blancas para convertir esos píxeles en listas de coordenadas geométricas
    #Los contornos delimitan cada objeto                                                   # RETR_EXTERNAL: contornos externos / #CHAIN_APPROX_SIMPLE: guarda el punto inicial y final

    # TRACKING (Seguimiento)
    blobs_actuales = [] # guardamos los coches detectados en el frame actual
    for cnt in contours: #cnt: coche posible
        # Filtrado por área
        if cv2.contourArea(cnt) >= AREA_MIN: # calculamos el área de cada contorno. Si es menor que 600 pixeles, lo descartamos
            (x, y, w_b, h_b) = cv2.boundingRect(cnt) # rectángulo que encierra el contorno
            #x,y: esquina superior izquierda
            # w_b: ancho
            # h_b: alto
            cx = x + w_b // 2 # centro X
            cy = y + h_b // 2 # centro Y
            blobs_actuales.append({'centro': (cx, cy), 'rect': (x,y,w_b,h_b)}) # centro: posición para comparar con coches anteriores

    blobs_usados_idx = set() #guardar qué blobs ya hemos asignado a un coche existente

    # Actualizar coches que ya conocíamos
    for i, coche in enumerate(coches_en_escena): #iteramos por cada coche que ya estamos rastreando
        mejor_dist = MATCH_DIST #Cualquier blob más lejos que esto no se considera el mismo coche
        mejor_blob_idx = -1 #Índice del blob que mejor coincide con este coche (-1, no hay ninguno aún)

        for j, blob in enumerate(blobs_actuales):
            if j in blobs_usados_idx: continue #Si este blob ya ha sido asignado a otro coche, lo saltamos
            # Calculamos distancia al blob nuevo
            dist = euclid(coche.last_pos(), blob['centro']) # medir la cercanía entre la ultima posición conocida del coche y el nuevo blob
            if dist < mejor_dist: # si encontramos uno mas cercano que el anterior, guardamos su indice como mejor candidato
                mejor_dist = dist
                mejor_blob_idx = j
        
        if mejor_blob_idx != -1: # Si encontramos el mismo coche (detección cercana)
            blob = blobs_actuales[mejor_blob_idx] #Recuperamos el blob elegido (el más cercano).
            coche.update(blob['centro'][0], blob['centro'][1], blob['rect'])
            blobs_usados_idx.add(mejor_blob_idx) #Así ningún otro coche podrá usarlo
        else:
            coche.missed += 1 # Si no, sumamos falta

    # Crear coches nuevos (los blobs que sobraron)
    for j, blob in enumerate(blobs_actuales):
        if j not in blobs_usados_idx:
            nuevo_coche = Coche(next_id, blob['centro'][0], blob['centro'][1], blob['rect'])
            coches_en_escena.append(nuevo_coche)
            next_id += 1

    # Limpiar memoria (borrar los muy perdidos)
    coches_en_escena = [c for c in coches_en_escena if c.missed <= MAX_MISSED_FRAMES] # nos quedamos solo con los coches que no llevan demasiados frames perdidos

    # Dibujar rectángulos
    # ZONA ROJA (Ancho completo)
    cv2.rectangle(frame_out, (x_izq, y_arriba), (x_der, y_abajo), (0, 0, 255), 2)

    for coche in coches_en_escena: # para cada coche del frame actual
        (x, y, w_b, h_b) = coche.rect # extraemos las coordenadas del coche
        cv2.rectangle(frame_out, (x, y), (x + w_b, y + h_b), (0, 255, 0), 1) # pintar el recuadro de color verde
        
        # para el conteo: ¿el coche cruza la zona?
        if not coche.counted: # si no ha contado el coche
            cx, cy = coche.last_pos()
            # Si la altura Y está entre las líneas rojas
            if (cy > y_arriba and cy < y_abajo):
                final_count += 1 # contar el coche cuando está dentro de la región
                coche.counted = True 

    cv2.imshow("Detector de Coches", frame_out)     # Mostrar video final
    
    if cv2.waitKey(30) & 0xFF == 27: # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

print(f"Contador final de coches: {final_count}")