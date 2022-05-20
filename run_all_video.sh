#!/bin/bash
cd src/
for filename in ../input/*.mp4; do
    name=${filename##*/}
    base=${name%.mp4}

    python detect_video_with_model.py -i $base
    
done
cd ..