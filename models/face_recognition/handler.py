import cv2
from interfaces import FaceRecognitionHandler, FaceRecognitionModel

class OpenCVFaceRecognition(FaceRecognitionHandler):
    def __init__(self, models: [FaceRecognitionModel]):
        self.models = models
        super().__init__(models)

    def draw_face_rectangles(self, image, faces, color):
        if color is None:
            color = (0, 255, 0)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    def handle_camera(self, cap):
        while True:
            ret, frame = cap.read()
            if ret:
                for model in self.models:
                    faces = model.detect_faces(frame)
                    self.draw_face_rectangles(frame, faces, model.color)
                cv2.imshow('Camera Face Detection', frame)
                cv2.setWindowProperty('Camera Face Detection', cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27 or cv2.getWindowProperty('Camera Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                    break
        cap.release()
        cv2.destroyAllWindows()

    def handle_image(self, image_path, width, height):
        image = cv2.imread(image_path)
        for model in self.models:
            faces = model.detect_faces(image)
            self.draw_face_rectangles(image, faces, model.color)

        if image.shape[1] > width or image.shape[0] > height:
            r = min(width / image.shape[1], height / image.shape[0])
            dim = (int(image.shape[1] * r), int(image.shape[0] * r))
            image = cv2.resize(image, dim)

        cv2.imshow('Image Face Detection', image)
        cv2.setWindowProperty('Image Face Detection', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('Image Face Detection', 0, 0)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27 or cv2.getWindowProperty('Image Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()

        return image

    def handle_video(self, video_path, width, height):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read() # Read frame from video
            if not ret:
                break

            # Calculate the ratio of the new height to the old height and resize the frame
            r = min(width / frame.shape[1], height / frame.shape[0])
            dim = (int(frame.shape[1] * r), int(frame.shape[0] * r))
            frame = cv2.resize(frame, dim)

            for model in self.models:
                faces = model.detect_faces(frame)
                self.draw_face_rectangles(frame, faces, model.color)
            cv2.imshow('Video Face Detection', frame)
            cv2.setWindowProperty('Video Face Detection', cv2.WND_PROP_TOPMOST, 1)

            # Move
            cv2.moveWindow('Video Face Detection', 0, 0)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or cv2.getWindowProperty('Video Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()