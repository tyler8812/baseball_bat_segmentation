
VIDEONAME = "C0036-24"
INPUTFOLDER = "input/"
OUTPUTFOLDER = "output/"
SAVE = False

def video_capture(video_name, frame_rate=20, width=1920, height=1080):
    import cv2
    print("reading " + "../" + INPUTFOLDER + video_name + ".mp4")
    cap = cv2.VideoCapture("../" + INPUTFOLDER + video_name + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        "./" + OUTPUTFOLDER + video_name + "_segment.mp4", fourcc, frame_rate, (width, height)
    )
    return cap, output



if __name__  =="__main__":

    from detectron2.utils.logger import setup_logger
    setup_logger()

    import cv2

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog


    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    cap, output_video = video_capture(VIDEONAME)
    while cap.isOpened():

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        outputs = predictor(frame)

        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        # special classes
        # 34: baseball bat
        outputs["instances"] = outputs["instances"][outputs["instances"].pred_classes == 34]

        # We can use `Visualizer` to draw the predictions on the image.# We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        output_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_frame = output_frame.get_image()[:, :, ::-1]
        cv2.imshow("image", output_frame)
        if SAVE:
            output_video.write(output_frame)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
