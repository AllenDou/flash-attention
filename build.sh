#
MAX_JOBS=16 NVCC_THREADS=1 HEADDIM=64 DTYPE=fp16 ENABLE_SM90=FALSE time pip install -e . -v
#/usr/bin/python3 /usr/local/bin/pytest  tests/test_flash_attn.py -k test_flash_attn_kvcache -s
