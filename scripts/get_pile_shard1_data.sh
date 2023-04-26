#!/bin/bash
# This script downloads the Pile shard 1 data and puts it under data/pile_01.
wget https://the-eye.eu/public/AI/pile/train/01.jsonl.zst
mkdir -p ../data
mkdir -p ../data/pile_01
mv 01.jsonl.zst ../data/pile_01
