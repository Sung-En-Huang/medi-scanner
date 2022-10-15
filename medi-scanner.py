import numpy as np
import cv2

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    ctime = 0

    while True:
        success , img = cap.read()
        img_iso = np.empty(img.shape)
        img_iso.fill(0)

        cv2.imshow("Capture" , img)

        if cv2.waitKey(1) &0xFF == ord('x'):
            break # Press 'x' to close window

if __name__ == '__main__':
    main()