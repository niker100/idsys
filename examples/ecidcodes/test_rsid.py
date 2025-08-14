import numpy as np
from ecidcodes.idcodes_rsid import RSID_U8

rsid = RSID_U8()
rsid.generate_gf_outer(8)
print(rsid.get_exp_arr())
print(rsid.get_log_arr())

# Define parameters
message = [1, 2, 3, 4]
tag_pos = 5
gf_exp = 8

exp_arr = rsid.get_exp_arr()
log_arr = rsid.get_log_arr()

result = rsid.rsid(message, tag_pos, exp_arr, log_arr, gf_exp)
print(f"RSID result: {result}")
