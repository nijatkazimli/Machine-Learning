import cv2

class FaceRecognition(FaceRecognitionHandler):
    def __init__(self, model: FaceRecognitionModel):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        super().__init__(model)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return faces

    def draw_face_rectangles(self, image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def handle_camera(self, cap):
        while True:
            ret, frame = cap.read()
            if ret:
                faces = self.detect_faces(frame)
                self.draw_face_rectangles(frame, faces)
                cv2.imshow('Camera Face Detection', frame)
                cv2.setWindowProperty('Camera Face Detection', cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27 or cv2.getWindowProperty('Camera Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                    break
        cap.release()
        cv2.destroyAllWindows()

    def handle_image(self, image_path, width, height):
        image = cv2.imread(image_path)
        faces = self.detect_faces(image)
        self.draw_face_rectangles(image, faces)
        if image.shape[1] > width or image.shape[0] > height:
            # Resize the image only if it is bigger than the screen size
            image = cv2.resize(image, (width, height))    
        cv2.imshow('Image Face Detection', image)
        cv2.setWindowProperty('Image Face Detection', cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27 or cv2.getWindowProperty('Image Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()

    def handle_video(self, video_path, width, height):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read() # Read frame from video
            if not ret:
                break

            # Resize frame to fit screen
            frame = cv2.resize(frame, (width, height))

            faces = self.detect_faces(frame)
            self.draw_face_rectangles(frame, faces)
            cv2.imshow('Video Face Detection', frame)
            cv2.setWindowProperty('Video Face Detection', cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or cv2.getWindowProperty('Video Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()