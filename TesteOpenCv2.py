import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializa a detecção de mãos
hands = mp_hands.Hands()

# Abre a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Converta a imagem para o formato BGR para processamento
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta mãos na imagem
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Desenha os pontos das mãos na imagem
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Identifica gestos básicos
            thumb_x = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_y = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_x = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            if thumb_x < index_x:
                gesture = "Polegar para cima"
            else:
                gesture = "Polegar para baixo"

            # Exibe o gesto na tela
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe o vídeo com as detecções
    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
