set -x
curl https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz > /tmp/aclImdb_v1.tar.gz
gunzip -f /tmp/aclImdb_v1.tar.gz
tar -xvf /tmp/aclImdb_v1.tar -C /tmp/
