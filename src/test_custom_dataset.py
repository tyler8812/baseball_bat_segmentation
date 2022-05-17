
if __name__=="__main__":
    import os, cv2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from utils import get_baseball_bat_dicts, register
    from detectron2.utils.visualizer import Visualizer
    from detectron2 import model_zoo
    
    baseball_bat_metadata = register()

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.OUTPUT_DIR = "../model"
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_baseball_bat_dicts("baseball_bat/test")
    for d in dataset_dicts:    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=baseball_bat_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_frame = out.get_image()[:, :, ::-1]
        cv2.imshow("frame", out_frame)
        if cv2.waitKey(0) == ord("q"):
            break
