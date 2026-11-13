import cv2
import numpy as np

filenames = ["images/0.jpg", "images/1.jpg", "images/2.jpg", "images/3.jpg", "images/4.jpg", "images/5.jpg"]

def captureImages(frame, key):
    if ord('1') <= key <= ord('6'):
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)

        b, g, r = cv2.split(blurred)

        lap_b = cv2.convertScaleAbs(cv2.Laplacian(b, cv2.CV_64F, ksize=5))
        lap_g = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F, ksize=5))
        lap_r = cv2.convertScaleAbs(cv2.Laplacian(r, cv2.CV_64F, ksize=5))

        combined_edges = np.maximum(lap_b, np.maximum(lap_g, lap_r))

        _, binary_edges = cv2.threshold(combined_edges, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = np.where(mask[:, :, None] == 255, frame, gray_3channel)

        cv2.imwrite(filenames[key - ord('0') - 1], result)
        print(f"Saved {filenames[key - ord('0') - 1]}")

# def processImages():
#     for imgPath in filenames:

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        exit()

    print("Video Capture satrted")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Webcam', frame)
            

        key = cv2.waitKey(1)
        captureImages(frame, key)

        if key == ord('6'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #processImages()

if __name__ == '__main__':
    main()
