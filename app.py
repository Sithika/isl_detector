from collections import deque
import cv2

cap = cv2.VideoCapture(0)


# Sentence buffer (last 30 gestures)
sentence_buffer = deque(maxlen=30)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks).reshape(1, -1)

                prediction = model.predict(landmarks)[0]
                gesture = label_encoder.inverse_transform([prediction])[0]

                # Avoid repeating same gesture too frequently
                if len(sentence_buffer) == 0 or sentence_buffer[-1] != gesture:
                    sentence_buffer.append(gesture)

        # Build a string from buffer
        full_sentence = " ".join([word for word in sentence_buffer if word])

        # Display the running sentence
        cv2.putText(frame, full_sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
