"""
This script uses Mesonet to predict the fakeness or non-fakeness of a given video

Basic Pipeline:
for each frame in video:
    1. extract face
    2. add it to a batch
    3. add original image to a batch
    
for each face in batch:
    1. Normalize images by dividing by 255.
    2. predict on face
    3. paste prediction on original image

return video
"""

import cv2
import matplotlib.pyplot as plt
import face_recognition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
import numpy as np
from classifiers import Meso4

# Parameters: --------------------------------------------------

# the video you want to configure
vid_path = '/home/aneeshj/CSecLab/deepfaketimit/DeepfakeTIMIT/' + \
        'lower_quality/fjre0/sa1-video-fcmh0.avi'

# the model you want to use (example: MesoInception, etc)
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# name of the output file
output_file_name = "predicted_video.avi"

# The padding around the face extracted.
# I haven't figured out how much of the face MesoNet really expects. If you want to manually increase the 
# padding around the `face_recognition` detected face, set this field. If you don't want padding, set it to 0
PADDING = 0

# The shape of the image sent to MesoNet. Should be 256x256 since MesoNet doesn't work on anything else
TARGET_IMAGE_SHAPE = (256, 256)

# Parameters for the text (no need to configure)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2




vidcap = cv2.VideoCapture(vid_path)
batch = [] 
batch_rgbs = []
frame_counter = 0

# read the first image and use it for shape determination
success, image = vidcap.read()

# Note: the image shapes have to be in reverse order (ie (image.shape[1], image.shape[0]))
#       not doing that can cause it to silently fail
out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (image.shape[1], image.shape[0]))


while(success):
    batch_rgbs.append(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=2)

    if len(boxes) == 0:
        print("No face we dectected in frame " + str(frame_counter) + ". Skipping it")
        success, image = vidcap.read()
        continue

    (top, right, bottom, left) = boxes[0]
    face = rgb[top - PADDING :bottom + PADDING, left - PADDING:right + PADDING]

    # if you want to see what images are fed to the model
    cv2.imwrite(str(frame_counter) + ".png", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    face = cv2.resize(face, TARGET_IMAGE_SHAPE)

    batch.append(face)
    success, image = vidcap.read()
    print("Finished processing frame " + str(frame_counter))

    frame_counter += 1

    # If you want to stop early, set the frame number below
    # if(frame_counter == 10):
    #     break


# Normalize the whole batch
batch = np.array(batch) / 255.

print("Batch shape is " + str(batch.shape))

predictions = classifier.predict(batch)


for img, predicted_value in zip(batch_rgbs, predictions):
    img = cv2.putText( img,'fakeness prob: ' + str(predicted_value[0]),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    out.write(img)

out.release()

print("The predictions are ", predictions)
print("The average prediction is ", np.mean(predictions))
