#!/bin/bash
# =============================================================================
# DATA SETUP for Xing et al. (2015) replication
# =============================================================================
# You need 3 things:
#   1. Source language embeddings (English)
#   2. Target language embeddings (Spanish)
#   3. Bilingual dictionary (train + test splits)
#
# Option A: Use MUSE data from Facebook (recommended, standardized)
# Option B: Use FastText pre-trained embeddings + your own dictionary
# =============================================================================

echo "=== Downloading MUSE bilingual dictionaries (EN-ES) ==="
# Train dictionary (~5000 pairs)
wget -q https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.0-5000.txt -O dict_en_es_train.txt
# Test dictionary (~1500 pairs)  
wget -q https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.5000-6500.txt -O dict_en_es_test.txt

echo "=== Downloading FastText pre-trained embeddings ==="
echo "WARNING: These files are ~2GB each. Download only what you need."
echo ""
echo "English (wiki.en.vec):"
echo "  wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
echo ""
echo "Spanish (wiki.es.vec):"
echo "  wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec"
echo ""
echo "=== Alternative: Use MUSE pre-aligned embeddings (smaller, 300d) ==="
echo "These are already aligned but useful for verification:"
echo "  wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.en.vec"
echo "  wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.es.vec"

echo ""
echo "=== Once downloaded, run: ==="
echo "python xing_orthogonal.py \\"
echo "    --src_emb wiki.en.vec \\"
echo "    --tgt_emb wiki.es.vec \\"
echo "    --dict_train dict_en_es_train.txt \\"
echo "    --dict_test dict_en_es_test.txt \\"
echo "    --max_vocab 200000 \\"
echo "    --lr 1.0 \\"
echo "    --n_iters 100"
