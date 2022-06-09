import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def rotate_img(img, angle, center_pos):
    h, w, _ = img.shape
    # center = (w // 2, h // 2) # 找到圖片中心
    center = center_pos
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))
    return rotate_img

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_detection = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_detection)
    if results.detections:
        for detection in results.detections:
            score = detection.score[0]
            xmin = int(frame.shape[1] * detection.location_data.relative_bounding_box.xmin)
            ymin = int(frame.shape[0] * detection.location_data.relative_bounding_box.ymin)
            width = int(frame.shape[1] * detection.location_data.relative_bounding_box.width)
            height = int(frame.shape[0] * detection.location_data.relative_bounding_box.height)
            left_top = [
                min(frame.shape[1] - 1, max(0, xmin)),
                min(frame.shape[0] - 1, max(0, ymin))
            ]
            right_bottom = [
                min(frame.shape[1] - 1, max(0, xmin+width)),
                min(frame.shape[0] - 1, max(0, ymin+height))
            ]
            # cv2.rectangle(frame, left_top, right_bottom, (0, 255, 0), 2)
            # for relative_keypoint in detection.location_data.relative_keypoints[:2]:
            #     pos = [int(frame.shape[1] * relative_keypoint.x), int(frame.shape[0] * relative_keypoint.y)]
            #     cv2.circle(frame, pos, 3, (255, 0, 0), -1)
            left_eye = [
                int(frame.shape[1] * detection.location_data.relative_keypoints[0].x),
                int(frame.shape[0] * detection.location_data.relative_keypoints[0].y)
            ]
            right_eye = [
                int(frame.shape[1] * detection.location_data.relative_keypoints[1].x),
                int(frame.shape[0] * detection.location_data.relative_keypoints[1].y)
            ]
            roll = int(math.atan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])) * 180 / math.pi)
            pos_roi_mid = (xmin + width // 2, ymin + height // 2)
            img_rotate = rotate_img(frame.copy(), roll, pos_roi_mid)
            # cv2.imshow('img_rotate', img_rotate)
            new_left_top = [
                min(left_top[0], right_bottom[0]),
                min(left_top[1], right_bottom[1])
            ]
            new_right_bottom = [
                max(left_top[0], right_bottom[0]),
                max(left_top[1], right_bottom[1])
            ]
            # cv2.rectangle(img_rotate, new_left_top, new_right_bottom, (0, 0, 255), 2)
            # cv2.imshow('img_rotate', img_rotate)
            img_roi = img_rotate.copy()[new_left_top[1]:new_right_bottom[1], new_left_top[0]:new_right_bottom[0]]
            # cv2.imshow('img_roi', img_roi)
            img_resize = cv2.resize(img_roi, (64, 64))
            print(f"{img_roi.shape} -> {img_resize.shape}")
            cv2.imshow('img_resize', img_resize)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
