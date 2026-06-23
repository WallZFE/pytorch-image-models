python validate.py  \
    --data-dir "../data/model_use/TUSimple" \
    --dataset "lane" \
    --split val \
    --model lane_net_18 \
    --workers 32 \
    --batch-size 32 \
    --input-size 3 288 640 \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --log-freq 50 \
    --checkpoint output/20260618-180219-lane_net_18-640/model_best.pth.tar \
    --num-gpu 1 \
    --no-prefetcher \
    --pin-mem \
    --results-file output/validate_results.csv \
    --num-classes 1000
    