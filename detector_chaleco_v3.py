"""
DETECTOR PRL v3 - Deteccion de Chaleco de Seguridad
Detecta si una persona lleva chaleco y envia alerta a Telegram si no lo lleva.
No se cierra - vuelve a preguntar en bucle.
Ventana de imagen pequena.
"""

import cv2
import requests
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# ============================================
# CONFIGURACION TELEGRAM
# ============================================
TELEGRAM_TOKEN = "8729863160:AAG9R0QRmhVJAJG7DILKJgEMqebO7flAxyQ"
CHAT_ID = "307270046"

# ============================================
# COLORES
# ============================================

VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
BLANCO = (255, 255, 255)

# Cargar modelo YOLO una sola vez
print("\n[INFO] Cargando modelo YOLO (solo la primera vez)...")
modelo = YOLO("yolo11n.pt")
print("[INFO] Modelo cargado correctamente!\n")


def enviar_alerta_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    datos = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        respuesta = requests.post(url, json=datos, verify=False)
        if respuesta.json().get("ok"):
            print("[TELEGRAM] Alerta enviada correctamente")
        else:
            print(f"[TELEGRAM] Error: {respuesta.json().get('description')}")
    except Exception as e:
        print(f"[TELEGRAM] Error de conexion: {e}")


def detectar_chaleco_por_color(imagen, bbox):
    x1, y1, x2, y2 = [int(c) for c in bbox]

    alto_persona = y2 - y1
    y_top = y1 + int(alto_persona * 0.15)
    y_bot = y1 + int(alto_persona * 0.65)
    recorte = imagen[y_top:y_bot, x1:x2]

    if recorte.size == 0:
        return False, 0

    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    lower_amarillo = np.array([20, 100, 100])
    upper_amarillo = np.array([45, 255, 255])

    lower_verde = np.array([35, 100, 100])
    upper_verde = np.array([85, 255, 255])

    lower_naranja = np.array([5, 100, 100])
    upper_naranja = np.array([25, 255, 255])

    mask_amarillo = cv2.inRange(hsv, lower_amarillo, upper_amarillo)
    mask_verde = cv2.inRange(hsv, lower_verde, upper_verde)
    mask_naranja = cv2.inRange(hsv, lower_naranja, upper_naranja)

    mask_total = mask_amarillo | mask_verde | mask_naranja

    total_pixeles = mask_total.size
    pixeles_fluorescentes = cv2.countNonZero(mask_total)
    porcentaje = (pixeles_fluorescentes / total_pixeles) * 100

    tiene_chaleco = porcentaje > 15

    return tiene_chaleco, porcentaje


def analizar_imagen(ruta_imagen):
    print(f"\n{'='*50}")
    print(f"  ANALIZANDO IMAGEN")
    print(f"{'='*50}")
    print(f"  Archivo: {ruta_imagen}")
    print(f"  Hora: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}\n")

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("[ERROR] No se pudo cargar la imagen")
        return

    print("[INFO] Analizando imagen...")
    resultados = modelo(imagen, verbose=False)

    personas_detectadas = 0
    personas_sin_chaleco = 0
    personas_con_chaleco = 0

    for resultado in resultados:
        for box in resultado.boxes:
            if int(box.cls[0]) == 0:
                personas_detectadas += 1
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(c) for c in bbox]

                tiene_chaleco, porcentaje = detectar_chaleco_por_color(imagen, bbox)

                if tiene_chaleco:
                    personas_con_chaleco += 1
                    color = VERDE
                    etiqueta = f"CON CHALECO ({porcentaje:.0f}%)"
                    print(f"  [OK] Persona {personas_detectadas}: CON chaleco (color: {porcentaje:.1f}%)")
                else:
                    personas_sin_chaleco += 1
                    color = ROJO
                    etiqueta = f"SIN CHALECO!"
                    print(f"  [!!] Persona {personas_detectadas}: SIN chaleco (color: {porcentaje:.1f}%)")

                cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(imagen, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(imagen, etiqueta, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLANCO, 2)

    print(f"\n{'='*50}")
    print(f"  RESULTADO:")
    print(f"  Personas detectadas: {personas_detectadas}")
    print(f"  Con chaleco: {personas_con_chaleco}")
    print(f"  Sin chaleco: {personas_sin_chaleco}")
    print(f"{'='*50}\n")

    if personas_sin_chaleco > 0:
        ahora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        mensaje = (
            f"\U0001f6a8 *ALERTA PRL \u2014 SIN CHALECO*\n\n"
            f"\u26a0\ufe0f Se ha detectado *{personas_sin_chaleco} persona(s)* "
            f"SIN chaleco de seguridad.\n\n"
            f"\U0001f4cb *Detalles:*\n"
            f"\u2022 Personas totales: {personas_detectadas}\n"
            f"\u2022 Con chaleco: {personas_con_chaleco}\n"
            f"\u2022 Sin chaleco: {personas_sin_chaleco}\n\n"
            f"\U0001f550 {ahora}\n"
            f"\U0001f916 _Detector PRL Automatico_"
        )
        enviar_alerta_telegram(mensaje)
    elif personas_detectadas > 0:
        print("[OK] Todas las personas llevan chaleco. No se envia alerta.")
    else:
        print("[INFO] No se detectaron personas en la imagen.")

    nombre_salida = "resultado_" + os.path.basename(ruta_imagen)
    cv2.imwrite(nombre_salida, imagen)
    print(f"[INFO] Imagen guardada: {nombre_salida}")

    # Ventana pequena
    imagen_pequena = cv2.resize(imagen, (640, 480))
    cv2.imshow("Detector PRL - Chaleco de Seguridad", imagen_pequena)
    print("[INFO] Pulsa cualquier tecla en la ventana para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================
# BUCLE PRINCIPAL - NO SE CIERRA
# ============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  DETECTOR PRL v3.0")
    print("  Deteccion de Chaleco de Seguridad")
    print("  Escribe 'salir' para cerrar")
    print("="*50)

    while True:
        print("\n  Escribe la ruta de la imagen a analizar:")
        print("  (Arrastra la imagen a esta ventana)")
        print("  (Escribe 'salir' para cerrar)\n")
        ruta = input("  Ruta: ").strip().strip('"').strip("'")

        if ruta.lower() == "salir":
            print("\n  Hasta luego! Detector PRL cerrado.")
            break

        if not ruta:
            print("[ERROR] No escribiste ninguna ruta. Intentalo de nuevo.")
            continue

        if not os.path.exists(ruta):
            print(f"[ERROR] No se encontro: {ruta}")
            print("  Verifica el nombre y la ruta. Intentalo de nuevo.")
            continue

        analizar_imagen(ruta)
