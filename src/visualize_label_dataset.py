

if __name__ == "__main__":
    
    import cv2
    from utils import get_baseball_bat_dicts, register
    from detectron2.utils.visualizer import Visualizer
    

    
    baseball_bat_metadata = register()

    dataset_dicts = get_baseball_bat_dicts("baseball_bat/test")

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=baseball_bat_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        out_frame = out.get_image()[:, :, ::-1]
        cv2.imshow("frame", out_frame)
        if cv2.waitKey(0) == ord("q"):
            break
