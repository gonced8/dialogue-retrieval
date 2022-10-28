#!/bin/bash

INDEX_DIRECTORY="data/multiwoz/index/bm25"

# Generate documents
python -m other.generate_bm25_documents --index_directory $INDEX_DIRECTORY

# Build index
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $INDEX_DIRECTORY/collection \
  --index $INDEX_DIRECTORY \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# Remove auxiliary documents
rm -rf $INDEX_DIRECTORY/collection
