from mediapipe import solutions
import cv2
import numpy as np

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, scaler, classifier, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.scaler = scaler
        self.classifier = classifier
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mp_hands = solutions.hands
        self.mp_drawing = solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        
        self.lmList = []

    def classify_hand_to_letter(self, img, draw=True, flipType=True):
        """
        detect hands from image and extract keypoints
        :param img: Image to find the hands from.
        :param draw: Flag to draw to output on the image.
        :return:Image with or without the interpreted letter
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = imgRGB.shape
        self.results = self.hands.process(imgRGB)
        allHands = []
        if self.results.multi_hand_landmarks:
            for handType, landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                landmark_list = []
                xList, yList = [], []
                for lm in landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])
                    xList.append(int(lm.x * width))
                    yList.append(int(lm.y * height))

                ## classify hand from landmarks
                predLabel = predict_landmark(landmark_list, self.scaler, self.classifier)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                            bbox[1] + (bbox[3] // 2)
                
                myHand["center"] = (cx, cy)
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)
                myHand['label'] = predLabel

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)


                ## draw
                if draw:
                    self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                    (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                    (255, 0, 255), 2)
                    # cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                    #                 2, (255, 0, 255), 2)
                    cv2.putText(img, myHand["label"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
                
        return allHands, img

def normalize_landmarks(FSLData):
    FSLData_normalized = (FSLData.copy()).reshape(-1,21,3)
    for idx, data in enumerate(FSLData):
        data = data.reshape(21,3)
        first_row = np.array(data[0].copy())
        FSLData_normalized[idx] = [np.abs(first_row - np.array(data[i])).tolist() for i in range(len(data))]
    return FSLData_normalized.reshape(-1, 21*3)

def predict_landmark(landmark_list, scaler, classifier):
    """ 
    Prepare landmarks before classification (normalize -> scale -> classify)
    :param landmark_list: List of landmarks of hand with shape (21,3)
    :scaler: Standard Scaler
    :classifier: Machine Learning classifier
    :return: Classified letter
    """
    landmark_list = np.array(landmark_list).reshape(1,63)
    # landmark_list = normalize_landmarks(landmark_list)
    test_data = scaler.transform(landmark_list)
    prediction = classifier.predict(test_data)
    return prediction[0]



