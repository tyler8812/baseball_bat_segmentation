# Program To Read video
# and Extract Frames
import cv2
import os
import shutil

# import sys

def FrameCapture(videoName, video_folder):

    videoPath = video_folder + "/" + videoName + ".mp4"
    frameDesFolder = "../video_frame" + "/" + videoName

    print("playing {}".format(videoPath))
    # Path to video file
    videoObj = cv2.VideoCapture(videoPath)
    # Used as counter variable
    count = 0

    # checks folder exist
    if os.path.isdir(frameDesFolder):
        shutil.rmtree(frameDesFolder)
    os.mkdir(frameDesFolder)

    success, frame = videoObj.read()
    while success:

        # Saves the frames with frame-count

        cv2.imwrite(frameDesFolder + "/" +
                    "{0:0=12d}.png".format(count), frame)
        # cv2.imshow("Frame", frame)
        count += 1

        # pressedKey = cv2.waitKey(1) & 0xFF
        # if pressedKey == ord("q"):

        #     break

        # read frame
        success, frame = videoObj.read()


# Driver Code
if __name__ == "__main__":

    # videoFolder = "719883"

    # checks folder exist
    # if os.path.isdir("videoFrame" + "/" + videoFolder):
    #     shutil.rmtree("videoFrame" + "/" + videoFolder)
    # os.mkdir("videoFrame" + "/" + videoFolder)

    video_folder = "../different_fps_video"
    for video in os.listdir(video_folder + "/"):
        videoName, extension = os.path.splitext(video)
        print(videoName, extension)
        # Calling the function
        if extension == ".mp4":
            FrameCapture(videoName, video_folder)
