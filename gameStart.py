import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw_landmarks_on_image import draw_landmarks_on_image
import os

# 손 인식 개수
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

#cap = cv2.VideoCapture(0)

#if not cap.isOpened():
#    print("웹캠을 연결할 수 없습니다")
#    exit()

# 입구 영역
start_rect = { "x1": 120, "y1": 900, "x2": 130, "y2": 1290}

# 출구 영역
exit_rect = {"x1": 2038, "y1": 900, "x2": 2045, "y2": 1290}

    
def play_game(level, count, cap):
    
    game_started = False
    state = "STOP"

    elapsed_time = 0
    pause_start_time = None
    # 게임 시작 전 점 위치 고정

   # 빨간 점의 초기 고정 위치 (입구 start_rect의 한가운데 좌표)
    start_center_x = (start_rect["x1"] + start_rect["x2"]) // 2  # 125
    start_center_y = (start_rect["y1"] + start_rect["y2"]) // 2  # 1095
    
    dot_x = start_center_x
    dot_y = start_center_y

    prev_index_x = None
    prev_index_y = None
    
    if level == 1:
        MAZE_PATH = "easy1.jpg"
    elif level == 2:
        MAZE_PATH = "medium1.jpg"
    elif level == 3:
        MAZE_PATH = "hard1.jpg"  

    print("현재 실행 위치:", os.getcwd())
    print("찾는 파일:", MAZE_PATH)
    print("파일 존재 여부:", os.path.exists(MAZE_PATH))

    base_maze_img = cv2.imread(MAZE_PATH)  # 미로 이미지 파일 경로
    game_time = time.time()
    last_hit_time = 0  # 마지막으로 검은색에 닿은 시간
    is_hit = False  # 검은색에 닿았는지 여부를 추적하는 변수

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다")
            break

        # OpenCV로 넘어오는 실시간 frame을 mp가 읽을 수 있게 변환
        # OpenCV (BGR) Channel
        # MP (RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        detection_result = detector.detect(image)

        # 검지 끝 좌표 가져오기
        index_x = None
        index_y = None
        is_pointer_gesture = False  # 검지 끝이 인식되었는지 여부를 추적하는 변수

        if detection_result.hand_landmarks:
            hand_landmark = detection_result.hand_landmarks[0]

            # 검지 끝 (8번)
            index_tip = hand_landmark[8]

            h, w, _ = frame.shape

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            joint_y = [landmark.y for landmark in hand_landmark]
            index_open  = joint_y[8]  < joint_y[6]   # 검지
            middle_open = joint_y[12] < joint_y[10]  # 중지
            ring_open   = joint_y[16] < joint_y[14]  # 약지
            pinky_open  = joint_y[20] < joint_y[18]  # 소지


            if index_open and not middle_open and not ring_open and not pinky_open:
                is_pointer_gesture = True    

        cvt_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
        annotated_image = draw_landmarks_on_image(cvt_image, detection_result)
        annotated_image = cv2.flip(annotated_image, 1)
        #cv2.imshow("FRAME", annotated_image)


        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #   break

        maze_img = base_maze_img.copy()  # 미로 이미지 파일 경로
        
        # 입구 표시 (파란색)
        cv2.rectangle(maze_img, (start_rect["x1"], start_rect["y1"]), (start_rect["x2"], start_rect["y2"]), (255,0,0), -1)
        
        
        # 출구 표시 (초록색)
        cv2.rectangle(maze_img, (exit_rect["x1"], exit_rect["y1"]), (exit_rect["x2"], exit_rect["y2"]), (0,255,0), -1)

        #이미지 2개 띄우기
        img_height = 480
        height_frame, width_frame = frame.shape[:2]
        height_maze, width_maze = maze_img.shape[:2]

        block_x = dot_x
        block_y = dot_y
                
        # 미로 이미지 크기에 맞게 좌표 변환
        if index_x is not None:
            maze_x = int((width_frame -index_x) * (width_maze / width_frame))
            maze_y = int(index_y * (height_maze / height_frame))
            
            if not game_started:
                
                dot_x = start_center_x
                dot_y = start_center_y

                if (start_rect["x1"] <= maze_x <= start_rect["x2"] and 
                    start_rect["y1"] <= maze_y <= start_rect["y2"]):
            
                    game_started = True
                    state = "START"
            
                    game_time = time.time()
            
                    print("게임 시작!")

                prev_index_x = index_x
                prev_index_y = index_y   

            else:
                if is_pointer_gesture:

                    # STOP -> START
                    if state == "STOP":
                        state = "START"

                        if pause_start_time is not None:
                            game_time += time.time() - pause_start_time
                    
                        print("시작")

                    if prev_index_x is not None and prev_index_y is not None:
                        dx = index_x - prev_index_x
                        dy = index_y - prev_index_y
                    
                        # 이동 속도 조절
                        speed = 1.5  # 속도 조절 인자 (1.5보다 크면 더 빠르게, 1.5보다 작으면 더 느리게)
                        block_x = max(0, min(dot_x - int(dx * speed), width_maze - 1))
                        if not np.all(maze_img[dot_y, block_x] < 30):
                            dot_x = block_x
                    
                        block_y = max(0, min(dot_y + int(dy * speed), height_maze - 1))
                        if not np.all(maze_img[block_y, dot_x] < 30):
                            dot_y = block_y
                    prev_index_x = index_x
                    prev_index_y = index_y

                            # 검은색에 닿았을 때는 이동하지 않음

                else :    
                    if state == "START":
                        state = "STOP"

                        pause_start_time = time.time()  # 일시정지 시작 시간 기록dt
                        print("정지")

                    prev_index_x = None
                    prev_index_y = None 

            if state == "START":                
                pixel = maze_img[block_y, block_x]
                
                #  빨간 점 좌표값이 #000000(검은색)에 닿으면 count 감소
                if np.all(pixel < 30):
                    current_time = time.time() # 현재 시간 가져오기
                    
                    # 벽에 닿았을 때 1초 이상 지났으면 count 감소
                    # 마지막으로 검은색에 닿은 시간과 현재 시간을 비교하여 1초 이상 지났으면 count 감소
                    if not is_hit and current_time - last_hit_time >= 1:  # 검은색에 처음 닿았을 때만 count 감소
                        count -= 1
                        is_hit = True
                        last_hit_time = current_time  # 마지막으로 검은색에 닿은 시간을 지금으로 업데이트
                        if count <= 0:
                            return "game_over"
                else:   
                    is_hit = False  # 검은색에서 벗어나면 다시 False로 설정
            
        
                if (exit_rect["x1"] <= dot_x <= exit_rect["x2"] and exit_rect["y1"] <= dot_y <= exit_rect["y2"]):
                    print("게임 클리어!")
                    cv2.destroyWindow("Maze_Game")
                    return "clear"
  
        else:
            prev_index_x = None
            prev_index_y = None
            
        cv2.circle(maze_img, (dot_x, dot_y), 20, (0, 0, 255), -1)  # 점 빨간색 원으로 표시    
        # 출구 도착 체크  
        # 이미지 크기 조정
        frame_resize = cv2.resize(annotated_image, (int(width_frame * (img_height / height_frame)), img_height))
        maze_resize = cv2.resize(maze_img, (int(width_maze * (img_height / height_maze)), img_height))
        
        # 두 이미지를 가로로 붙이기
        combine_img = np.hstack((frame_resize, maze_resize))
				
        if state == "START":
            elapsed_time = time.time() - game_time
        
        cv2.putText(combine_img, f"State : {state}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(combine_img, f"Life : {count}", (230,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(combine_img, f"Level: {level}", (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combine_img, f"Time : {elapsed_time:.1f}", (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Maze_Game", combine_img,)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows("Maze_Game")
            return "game_over"
