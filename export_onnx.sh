python onnx_export.py ./output/output_model.onnx \
    --model lane_net_18 \
    --batch-size 1 \
    --input-size 3 288 640 \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --checkpoint output/20260618-180219-lane_net_18-640/last.pth.tar