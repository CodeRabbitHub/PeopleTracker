import cv2


class VideoHandler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.resolution = (
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video.release()

    def get_video_frames_generator(self, stride=1):
        # Open the video file
        video = cv2.VideoCapture(self.video_path)

        # Iterate over the frames in the video
        frame_index = 0
        while True:
            # Read the next frame
            ret, frame = video.read()

            # If the frame cannot be read, we've reached the end of the video
            if not ret:
                break

            # Yield the frame based on the specified stride
            if frame_index % stride == 0:
                yield frame

            frame_index += 1

        # Release the video capture object
        video.release()
