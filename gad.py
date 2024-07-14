import cv2  # Import OpenCV library for computer vision tasks
import math  # Import math library (though it's not used in this script)
import argparse  # Import argparse library for command-line argument parsing

# Function to detect faces in a frame and highlight them
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()  # Make a copy of the input frame
    frameHeight = frameOpencvDnn.shape[0]  # Get the height of the frame
    frameWidth = frameOpencvDnn.shape[1]  # Get the width of the frame
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)  # Set the blob as input to the network
    detections = net.forward()  # Perform a forward pass to get detections
    faceBoxes = []  # Initialize an empty list to store face bounding boxes
    # Iterate over all detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get the confidence score of the detection
        if confidence > conf_threshold:  # Check if the confidence is above the threshold
            # Calculate the coordinates of the bounding box
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # Append the bounding box to the list
            # Draw a rectangle around the detected face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes  # Return the frame with highlighted faces and the list of bounding boxes


# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image')  # Add an optional argument for the input image

args = parser.parse_args()  # Parse the command-line arguments

# File paths for the face detection model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
# File paths for the age detection model files
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
# File paths for the gender detection model files
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values for the age and gender models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# List of age ranges corresponding to the age model output
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# List of gender labels corresponding to the gender model output
genderList = ['Male', 'Female']

# Load the face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Load the age detection model
ageNet = cv2.dnn.readNet(ageModel, ageProto)
# Load the gender detection model
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open a video capture object (image file or webcam)
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20  # Padding around the detected face for age and gender prediction
# Loop to read frames from the video capture
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()  # Read a frame from the video capture
    if not hasFrame:  # If no frame is read, break the loop
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)  # Detect and highlight faces
    if not faceBoxes:  # If no faces are detected, print a message
        print("No face detected")

    # Loop over the detected face bounding boxes
    for faceBox in faceBoxes:
        # Extract the face region with padding
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                     :min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Create a blob from the face region for gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)  # Set the blob as input to the gender network
        genderPreds = genderNet.forward()  # Perform a forward pass to get gender predictions
        gender = genderList[genderPreds[0].argmax()]  # Get the predicted gender
        print(f'Gender: {gender}')  # Print the predicted gender

        ageNet.setInput(blob)  # Set the blob as input to the age network
        agePreds = ageNet.forward()  # Perform a forward pass to get age predictions
        age = ageList[agePreds[0].argmax()]  # Get the predicted age range
        print(f'Age: {age[1:-1]} years')  # Print the predicted age range

        # Put the gender and age text on the result image
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)  # Show the result image with detected faces, genders, and ages
