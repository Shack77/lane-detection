import cv2
import numpy as np

def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(100, height), (image.shape[1] - 100, height), (image.shape[1] // 2, height // 2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left.append(parameters)
        else:
            right.append(parameters)
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    left_line = make_coordinates(image, left_avg)
    right_line = make_coordinates(image, right_avg)
    return np.array([left_line, right_line])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        canny = canny_edge(frame)
        cropped = region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        if lines is not None:
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        else:
            combo_image = frame
        cv2.imshow("Lane Detection", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("test_video.mp4")
