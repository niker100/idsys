## from validation.py:

````
gf_exp = 8
Generated 1000000 test messages
RSID Metrics:
  unique_tags: 256.0000
  tag_entropy: 7.9998
  tag_uniqueness: 0.0003
  tag_distribution_uniformity: 0.0002
RS2ID Metrics:
  unique_tags: 16.0000
  tag_entropy: 4.0000
  tag_uniqueness: 0.0000
  tag_distribution_uniformity: 0.0000
````

````
gf_exp = 16
Generated 200000 test messages
RSID Metrics:
  unique_tags: 62465.0000
  tag_entropy: 15.7436
  tag_uniqueness: 0.3123
  tag_distribution_uniformity: 0.1871
RS2ID Metrics:
  unique_tags: 256.0000
  tag_entropy: 7.9991
  tag_uniqueness: 0.0013
  tag_distribution_uniformity: 0.0009
````

- Within GF(2^8), RSID produces 2^8 = 256 unique tags as expected, whilst RS2ID only produces 2^4 = 16 ones. This is also true for gf_exp = 16 (2^16 = 65.536 unique tags for RSID, only 2^8 = 256 for RS2ID)

→ Implementation issue?