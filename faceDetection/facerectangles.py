import dlib
import cv2

def read_image(path):
    image = cv2.imread(path)
    return image

def check_rectangles(rectangles):
    return True if len(rectangles) == 1 else False

def landmark_points(img):
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    rectangles = detector(gray)
    #check if there is only one rectangle
    check_rectangles(rectangles)
    # Locate rectangle
    br_x = rectangles[0].br_corner().x
    br_y = rectangles[0].br_corner().y
    tl_x = rectangles[0].tl_corner().x
    tl_y = rectangles[0].tl_corner().y
    cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), 2)

def show_landmark_points(im_path):
    img = cv2.imread(im_path)
    landmark_points(img)
    cv2.imwrite("landmarks2.png", img)

show_landmark_points("face2.png")
