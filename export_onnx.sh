python onnx_export.py ./output/output_model.onnx \
    --model lane_net_regnetx_004 \
    --batch-size 1 \
    --input-size 3 288 640 \
    --mean 0.5 0.5 0.5 \
    --std 0.5 0.5 0.5 \
    --checkpoint output/20260815-180554-lane_net_regnetx_004-640/model_best.pth.tar