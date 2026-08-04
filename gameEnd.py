import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# 버튼 크기
BTN_WIDTH = 180
BTN_HEIGHT = 90

# 왼쪽 : 다시하기
restart_btn = {
    "x1": 100,
    "y1": 180,
    "x2": 100 + BTN_WIDTH,
    "y2": 180 + BTN_HEIGHT
}

# 오른쪽 : 종료
exit_btn = {
    "x1": 360,
    "y1": 180,
    "x2": 360 + BTN_WIDTH,
    "y2": 180 + BTN_HEIGHT
}

def game_end(cap, message="GAME OVER"):

    base_options = python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )

    detector = vision.HandLandmarker.create_from_options(options)

    selected_button = None
    select_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        index_x = None
        index_y = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            h, w, _ = frame.shape

            index_tip = hand[8]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)

        # 화면 그리기
        cv2.putText(frame, message, (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # 다시하기 버튼
        cv2.rectangle(frame, (restart_btn["x1"], restart_btn["y1"]), (restart_btn["x2"], restart_btn["y2"]), (255, 0, 0), 3)
        cv2.putText(frame, "RESTART", (125, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 종료 버튼
        cv2.rectangle(frame, (exit_btn["x1"], exit_btn["y1"]), (exit_btn["x2"], exit_btn["y2"]), (0, 0, 255), 3)
        cv2.putText(frame, "EXIT", (410, 235), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 손가락 버튼 판정
        if index_x is not None:
            current_button = None

            # 다시하기 버튼 위
            if (restart_btn["x1"] <= index_x <= restart_btn["x2"] and
                restart_btn["y1"] <= index_y <= restart_btn["y2"]):

                current_button = "restart"

            # 종료 버튼 위
            elif (exit_btn["x1"] <= index_x <= exit_btn["x2"] and
                exit_btn["y1"] <= index_y <= exit_btn["y2"]):

                current_button = "exit"

            # 버튼 위에 있을 때
            if current_button is not None:

                if selected_button != current_button:
                    # 처음 버튼 위에 올라온 순간
                    selected_button = current_button
                    select_time = time.time()

                else:
                    # 2초 유지
                    if time.time() - select_time >= 2:

                        if current_button == "restart":
                            cv2.destroyWindow("Game End")
                            return "restart"

                        elif current_button == "exit":
                            cv2.destroyAllWindows()
                            cap.release()
                            return "exit"

            # 버튼 밖으로 나가면 초기화
            else:
                selected_button = None
                select_time = None
              
        else:
            selected_button = None
            select_time = None
            
        if selected_button is not None:
            remain = max(0, 2 - (time.time() - select_time))

            cv2.putText(frame, f"{selected_button.upper()} : {remain:.1f}s", (250, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Game End", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            return "exit"
