INPUTFOLDER = "input/"
OUTPUTFOLDER = "different_fps_video/"
VIDEONAME = "C0042-12"


def video_capture(video_name, frame_rate=20, width=1920, height=1080):
    import cv2
    print("reading " + "../" + INPUTFOLDER + video_name + ".mp4")
    cap = cv2.VideoCapture("../" + INPUTFOLDER + video_name + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        "../" + OUTPUTFOLDER + video_name + ".mp4", fourcc, frame_rate, (width, height)
    )
    return cap, output

if __name__ == "__main__":
    import cv2
    cap, output_video = video_capture(VIDEONAME)
    save_in_every_n_frame = 7
    count = 1
    while cap.isOpened():

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if count % save_in_every_n_frame == 0:
            print(count)
            cv2.imshow("image", frame)
            output_video.write(frame)
        count += 1
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()