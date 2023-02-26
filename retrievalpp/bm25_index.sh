#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$1" \
  --index "$2" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw