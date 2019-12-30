import cv2

FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        # print(image.shape)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) > 0:
            x, y, w, h = rects[0]
            cv2.rectangle(image, (x,y),(x + w, y + h), (255,255,255), 3)
        # print(image.shape)
        # rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # print(rects)
        # data.update({"num_faces": len(rects), "faces": rects, "success": True})
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
