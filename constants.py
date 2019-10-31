import os
PATH_HOCKEY_FRAMES_VIOLENCE = '/media/david/datos/Violence DATA/HockeyFights/frames/violence'
PATH_HOCKEY_FRAMES_NON_VIOLENCE = '/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence'
PATH_VIOLENTFLOWS_FRAMES = '/media/david/datos/Violence DATA/violentflows/movies Frames'
PATH_CHECKPOINTS_DI = 'checkpoints/di'
PATH_CHECKPOINTS_MASK = 'checkpoints/masked'
PATH_LEARNING_CURVES_DI = 'learningCurves/di'
PATH_LEARNING_CURVES_MASK = 'learningCurves/masked'
PATH_SALIENCY_MODELS = 'Saliency/Models'
PATH_BLACK_BOX_MODELS = 'BlackBoxModels'
LABEL_PRODUCTION_MODEL = '-PRODUCT'
PATH_SALIENCY_DATASET = 'data/saliency'
PATH_UCFCRIME2LOCAL_VIDEOS = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'
PATH_UCFCRIME2LOCAL_FRAMES = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames'
PATH_UCFCRIME2LOCAL_README = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/readme'
PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/readme/Txt annotations'
PATH_UCFCRIME2LOCAL_FRAMES_REDUCED = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames_reduced'

ANOMALY_PATH_CHECKPOINTS = 'AnomalyCrime/checkpoints'
ANOMALY_PATH_LEARNING_CURVES = 'AnomalyCrime/learning_curves'
ANOMALY_PATH_TRAIN_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
ANOMALY_PATH_TEST_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
ANOMALY_PATH_SALIENCY_MODELS = 'saliencyModels/anomaly'
ANOMALY_PATH_BLACK_BOX_MODELS = 'BlackBoxModels/anomaly'