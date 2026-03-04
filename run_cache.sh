#!/bin/bash
cd /Users/mattrepp/Desktop/RL/ProspectRL
python -m prospect_rl.tools.cache_chunks \
    --world-path data/worlds/1 data/worlds/2 \
    --output-dir data/chunk_cache/combined \
    --chunk-size 64 384 64 \
    --y-min -64 --y-max 320 \
    --samples 9999 \
    --analyze \
    --seed 42 \
    2>&1 | tee /tmp/cache_chunks_output.log
echo "EXIT CODE: $?"
