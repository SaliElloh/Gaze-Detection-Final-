
import cv2
from config import PROCESS_EVERY_N_FRAMES
from fatigue_detection.eye_detector import EyeDet
from fatigue_detection.pose_estimation import HeadPoseEst
from fatigue_detection.attention_scorer import AttScorer
from fatigue_detection.alert_notifier import AlertNotifier

class FatiguePipeline:

    def __init__(self):
        self.eye_detector = EyeDet()
        self.pose_estimator = HeadPoseEst()
        self.attention_scorer = AttScorer()
        self.notifier = AlertNotifier()

    def process_frame(self, frame):
        eye_metrics = self.eye_detector.detect(frame)
        head_pose = self.pose_estimator.estimate(frame)
        attention = self.attention_scorer.score(eye_metrics, head_pose)

        return {
            "eye_metrics": eye_metrics,
            "head_pose": head_pose,
            "attention": attention
        }

    def handle_results(self, results):
        if results["attention"] < 0.4:
            self.notifier.alert_driver()


def main(video_path):

    cap = cv2.VideoCapture(video_path)
    pipeline = FatiguePipeline()

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        results = pipeline.process_frame(frame)
        pipeline.handle_results(results)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)
