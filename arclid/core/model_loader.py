from ultralytics import YOLO


def load_model():
    # out_path = args.out_path
    model_path_5k = '/zhome/55/8/198316/projects/variant-calling/run19_cutsize5k_resize1280_2models4HIFIandONT/ai_model_hifi/runs/detect/yolo11x_ccs_filtered_100perc_empty/weights/best.pt'
    model_path_50k = '/zhome/55/8/198316/projects/variant-calling/run20_cutesize50k/ai_model_hifi/runs/detect/yolo11x_ccs_filtered_100perc_empty/weights/best.pt'

    model_5k = YOLO(model_path_5k)
    model_50k = YOLO(model_path_50k)
    
    return model_5k, model_50k