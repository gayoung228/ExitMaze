import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw_landmarks_on_image import draw_landmarks_on_image

from gameStart import play_game
from level import detect_level
from gameEnd import game_end

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 연결할 수 없습니다")
    exit()
    
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다")
        break

    frame = cv2.flip(frame, 1)

    # 시작 버튼 영역 그리기
    start_x1 = 200
    start_y1 = 200
    start_x2 = 400
    start_y2 = 300

    cv2.rectangle(frame, (start_x1, start_y1), (start_x2, start_y2), (0,255,0), 3)
    cv2.putText(frame, "START", (230,260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    index_x = None
    index_y = None

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        index_tip = hand_landmarks[8]

        h, w, _ = frame.shape

        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)

        cv2.circle(frame, (index_x,index_y), 10, (0,0,255), -1)
        
    if index_x is not None:
        if (start_x1 < index_x < start_x2 and start_y1 < index_y < start_y2):
            print("START 클릭")

            # 시작 화면 종료
            cv2.destroyWindow("Main")

             # 레벨 선택 화면으로 이동
            level = detect_level(cap)

            if level is not None:
                cap.release()
                cv2.destroyAllWindows()

                game_cap = cv2.VideoCapture(0)
                result = play_game(level, count=10, cap=game_cap)

                end_result = None
                    
                if result == "game_over":
                    end_result = game_end(game_cap, "GAME OVER")

                elif result == "clear":
                    end_result = game_end(game_cap, "GAME CLEAR")

                if end_result == "restart":
                    game_cap.release()

                    # 처음(Start 화면)으로 돌아감
                    # while문을 다시 시작하도록 break 대신 continue 사용
                    cap = cv2.VideoCapture(0)
                    continue

                elif end_result == "exit":
                    game_cap.release()
                    break

    cv2.imshow("Main", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
