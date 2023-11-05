import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_face_rectangles(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("Select an option:")
print("1. Camera")
print("2. Image")
print("3. Video")
choice = input("Enter 1, 2, or 3: ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)
        draw_face_rectangles(frame, faces)
        cv2.imshow('Camera Face Detection', frame)
        cv2.setWindowProperty('Camera Face Detection', cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == '2':
    image_path = input("Enter the path to the image: ")
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    draw_face_rectangles(image, faces)
    cv2.imshow('Image Face Detection', image)
    cv2.setWindowProperty('Image Face Detection', cv2.WND_PROP_TOPMOST, 1)
    key = cv2.waitKey(0)
    if key == ord('q') or key == 27:
        cv2.destroyAllWindows()

elif choice == '3':
    video_path = input("Enter the path to the video: ")
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)
        draw_face_rectangles(frame, faces)
        cv2.imshow('Video Face Detection', frame)
        cv2.setWindowProperty('Video Face Detection', cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Please enter 1, 2, or 3.")