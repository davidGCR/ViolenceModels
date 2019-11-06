# # TEST
python3 AnomalyCrime/anomaly_main.py \
--operation testing \
--ndis 1 \
--testModelFile AnomalyCrime/checkpoints/resnet18_frames_Finetuned-False-_di-1_fusionType-tempMaxPool_num_epochs-22_videoSegmentLength-16_positionSegment-random-FINAL.pth

# # TRAIN-VALIDATION
# python3 AnomalyCrime/anomaly_main.py \
# --checkpointPath AnomalyCrime/checkpoints \
# --operation trainingFinal \
# --modelType resnet18 \
# --featureExtract true \
# --numEpochs 22 \
# --ndis 1 \
# --batchSize 8 \
# --numWorkers 4 \
# --shuffle true \
# --videoSegmentLength 16 \
# --positionSegment random

# SALIENCY
# python3 AnomalyCrime/saliencyAnomaly.py  \
# --batchSize 8 \
# --numEpochs 10 \
# --numWorkers 1  \
# --saliencyCheckout Saliency/Models/anomaly \
# --blackBoxFile AnomalyCrime/checkpoints/BlackBoxModels/resnet18_frames_Finetuned-False-_di-1_fusionType-tempMaxPool_num_epochs-15_videoSegmentLength-16_positionSegment-random.pth \
# --maxNumFramesOnVideo 0 \
# --videoSegmentLength 16 \
# --positionSegment random
