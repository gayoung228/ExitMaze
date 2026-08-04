import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

max_num_hands = 1 # 손 인식 개수

def detect_level(cap):
    gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
    }
    rps_gesture = {1: 1, 2: 2, 3: 3} # 1: easy, 9: medium, 3: hard

    # MediaPipe 1.0.0+ 제거된 mp.solutions.drawing_utils 대체용 관절 연결선
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8),           # 검지
        (5, 9), (9, 10), (10, 11), (11, 12),      # 중지
        (9, 13), (13, 14), (14, 15), (15, 16),    # 약지
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 새끼 & 손목
    ]

#    try:
#         file = np.genfromtxt('gesture_train.csv', delimiter=',')
#         angle = file[:, :-1].astype(np.float32) # 각도 데이터
#         label = file[:, -1].astype(np.float32)  # 라벨 데이터

#         knn = cv2.ml.KNearest_create()
#         knn.train(angle, cv2.ml.ROW_SAMPLE, label)
#         print("KNN 모델 학습 완료!")
#     except Exception as e:
#         print(f"gesture_train.csv 파일을 읽는 중 오류가 발생했습니다: {e}")
#        sys.exit() 

    # 3. MediaPipe Tasks API - HandLandmarker 설정 (VIDEO 모드)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # 실시간 비디오 스트리밍 모드
        num_hands=max_num_hands
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 4. 실시간 웹캠 제스처 인식 루프
    #cap = cv2.VideoCapture(0)

    #if not cap.isOpened():
    #    print("웹캠을 연결할 수 없습니다")
    #    sys.exit()

    selected_level = None
    select_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다")
            break

        # 1) MediaPipe에는 좌우 반전 안 한 원본 RGB 전달 (손 인식 정확도 유지)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # 2) Video 모드용 타임스탬프 전달 및 감지
        frame_timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # 시각화용 copy 생성 (RGB)
        annotated_img = np.copy(img_rgb)
        h, w, _ = annotated_img.shape

        # 3) 손을 인식했을 때 처리
        if result.hand_landmarks:
            for res in result.hand_landmarks:
                # 21개 랜드마크 x, y, z 추출
                joint = np.zeros((21, 3))
                points = []
                for j, lm in enumerate(res):
                    joint[j] = [lm.x, lm.y, lm.z]
                  
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))

                # --- 관절 각도 계산 (기존 알고리즘 유효) ---
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20, 3] 관절 벡터

                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 벡터 내적 후 arccos으로 각도 계산 [15,]
                calc_angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

                calc_angle = np.degrees(calc_angle) # Radian -> Degree

                # --- KNN 모델 추론 ---
                # data = np.array([calc_angle], dtype=np.float32)
                # ret_val, results, neighbours, dist = knn.findNearest(data, 3)
                # idx = int(results[0][0])

                #--- 손가락 개수 세기 ---
                index_finger = joint[8,1] < joint[6,1]  
                # 카메라에 손가락이 위쪽에 위치하면 True, 아니면 False
                middle_finger = joint[12,1] < joint[10,1]
                ring_finger = joint[16,1] < joint[14,1]
                finky_finger = joint[20,1] < joint[18,1]

                if index_finger and not middle_finger and not ring_finger and not finky_finger:
                    idx = 1  # 손가락 1개 (EASY)
                elif index_finger and middle_finger and not ring_finger and not finky_finger:
                    idx = 2  # 손가락 2개 (MEDIUM)
                elif index_finger and middle_finger and ring_finger and not finky_finger:
                    idx = 3  # 손가락 3개 (HARD)
                else :
                    idx = None # 기타 손 모양

                print("예측:", idx) #######

                # --- OpenCV 랜드마크 시각화 (Solutions 대체) ---
                # 1. 마디 연결선 그리기
                for p1, p2 in HAND_CONNECTIONS:
                    cv2.line(annotated_img, points[p1], points[p2], (200, 200, 200), 2)
                # 2. 관절 마디 점 그리기
                for pt in points:
                    cv2.circle(annotated_img, pt, 5, (255, 0, 0), -1)

                # --- 결과 텍스트 표시 ---
                if idx in rps_gesture:
                    current_level = rps_gesture[idx]

                    if selected_level != current_level:
                        selected_level = current_level
                        select_time = time.time()

                    else:
                        if time.time() - select_time >= 2:
                            cap.release()
                            cv2.destroyAllWindows()
                            return selected_level
                else: 
                    selected_level = None
                    select_time = None

        # 4) RGB -> BGR 변환 후 거울 모드(좌우 반전) 적용하여 출력
        annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        final_frame = cv2.flip(annotated_bgr, 1)

        # 레벨 선택 카운트다운 표시
        if selected_level is not None:
            remain = 2 - (time.time() - select_time)

            cv2.putText(final_frame, f"LEVEL {selected_level}", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(final_frame, f"{remain:.1f} 초 후에 시작 ", (40,270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        else:
            remain = 2
            cv2.putText(final_frame, f"LEVEL NONE", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(final_frame, f"{remain:.1f} 초 후에 시작 ", (40,270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # 레벨 선택 화면
        cv2.rectangle(final_frame, (20, 20), (330, 180), (50, 50, 50), -1)
        cv2.putText(final_frame, "SELECT LEVEL", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(final_frame, "1 Finger : EASY", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, "2 Fingers : MEDIUM", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, "3 Fingers : HARD", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Select Level', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return selected_level
