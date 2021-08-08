#!/bin/bash

listVar="RandomFlip RandomRotation RandomContrast RandomZoom RandomTranslation RandomCrop RandomFlip_prob RandomRotation_prob RandomTranslation_prob"
for i in $listVar; do
    echo "$i"
    python 3.py --key "$i"
done

