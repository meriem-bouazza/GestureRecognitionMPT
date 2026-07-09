import sys
sys.path.append(".")

import cv2
from hand_detector import HandDetector

detector = HandDetector()
detector.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result = detector.step(frame)

    if result["detected"]:
        color = (0, 255, 0)        
    else:
        color = (0, 0, 255)        

    h, w = frame.shape[:2]
    cv2.circle(frame, (w - 40, 40), 20, color, -1)
    
    cv2.imshow("HandDetector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.stop()
cv2.destroyAllWindows()