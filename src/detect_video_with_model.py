INPUTFOLDER = "input/"
OUTPUTFOLDER = "output/"
SAVE = True

def video_capture(video_name, frame_rate=20, width=1920, height=1080):
    import cv2
    print("reading " + "../" + INPUTFOLDER + video_name + ".mp4")
    cap = cv2.VideoCapture("../" + INPUTFOLDER + video_name + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        "../" + OUTPUTFOLDER + video_name + "_segment.mp4", fourcc, frame_rate, (width, height)
    )
    return cap, output



if __name__  =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument("-i", "--input_video", help="input video to run", default="C0036-24")


    args = parser.parse_args()
    
    from detectron2.utils.logger import setup_logger
    setup_logger()

    import cv2, os

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from utils import register
    from detectron2.utils.visualizer import ColorMode

    baseball_bat_metadata = register()
    
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.OUTPUT_DIR = "../model"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)


    cap, output_video = video_capture(args.input_video)
    while cap.isOpened():

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        outputs = predictor(frame)

        v = Visualizer(frame[:, :, ::-1],
                    metadata=baseball_bat_metadata, 
                     
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        output_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_frame = output_frame.get_image()[:, :, ::-1]
        cv2.imshow("image", output_frame)
        if SAVE:
            output_video.write(output_frame)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
