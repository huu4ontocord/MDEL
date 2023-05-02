#!/bin/bash
# This script downloads the Pile shard 1 data and puts it under data/pile_01.
mkdir -p ../data
mkdir -p ../data/pile
mkdir -p ../data/pile/train
mkdir -p ../data/pile/test
mkdir -p ../data/pile/val

wget https://the-eye.eu/public/AI/pile/train/01.jsonl.zst -P ../data/pile/train
wget https://the-eye.eu/public/AI/pile/test.jsonl.zst -P ../data/pile/test
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst -P ../data/pile/val
