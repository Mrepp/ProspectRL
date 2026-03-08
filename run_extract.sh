#!/bin/bash
cd /Users/mattrepp/Desktop/RL/ProspectRL

# Fix numpy version for amulet compatibility
pip install 'numpy~=1.26' 2>&1 | tail -3

# Verify amulet works
python -c "import amulet; print('amulet OK')" 2>&1

# Run extraction
python -m prospect_rl.tools.cache_chunks \
    --world-path data/worlds/1 data/worlds/2 \
    --output-dir data/chunk_cache/combined \
    --chunk-size 64 384 64 \
    --y-min -64 --y-max 320 \
    --samples 9999 \
    --analyze \
    --seed 42 \
    2>&1

echo "EXIT CODE: $?"
