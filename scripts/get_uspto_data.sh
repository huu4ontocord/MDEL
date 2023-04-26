#!/bin/bash
# This script downloads the USPTO Pile data and puts it under data/pile_uspto.
wget https://the-eye.eu/public/AI/pile_preliminary_components/pile_uspto.tar
tar -xf pile_uspto.tar
rm pile_uspto.tar
mkdir -p ../data
mv pile_uspto ../data
