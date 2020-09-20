# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy

# OpenCVのカスケードファイルと学習済みモデルのパスを指定
CASCADE_PATH = "./haarcascades/"
CASCADE = cv2.CascadeClassifier(
    CASCADE_PATH + 'haarcascade_frontalface_default.xml')

LEARNED_MODEL_PATH = "./learned-models/"
PREDICTOR = dlib.shape_predictor(
    LEARNED_MODEL_PATH + 'shape_predictor_68_face_landmarks.dat')

def face_position(gray_img):
    faces = CASCADE.detectMultiScale(gray_img, minSize=(100, 100))
    return faces

def facemark(gray_img):
    faces_roi = face_position(gray_img)
    landmarks = []

    for face in faces_roi:
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_img, 1)
        landmarks = []

        for rect in rects:
            landmarks.append(
                numpy.array([[p.x, p.y] for p in PREDICTOR(gray_img, rect).parts()]))
    print(landmarks[0])
    print(len(landmarks[0]))
    return landmarks


def main():
    img = cv2.imread("./img/input.jpg")  # 自分の画像に置き換え
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 処理を早くするためグレースケールに変換
    landmarks = facemark(gray)  # ランドマーク検出

    # ランドマークの描画
    for landmark in landmarks:
        for points in landmark:
            cv2.drawMarker(img, (points[0], points[1]), (21, 255, 12))

    # 保存
    cv2.imwrite("./result/output.jpg", img)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
