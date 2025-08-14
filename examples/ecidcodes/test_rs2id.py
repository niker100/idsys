import numpy as np
from ecidcodes.idcodes_rsid import RSID_U8
from ecidcodes.idcodes_rs2id import RS2ID_U8

rs2id = RS2ID_U8()
rs2id.generate_gf_outer(8)
rs2id.generate_gf_inner(8)
print(rs2id.get_exp_arr_in())
print(rs2id.get_log_arr_in())

# Define parameters
message = [1, 2, 3, 4]
tag_pos = 5
tag_pos_in = 5
gf_exp = 8

exp_arr = rs2id.get_exp_arr()
log_arr = rs2id.get_log_arr()
exp_arr_in = rs2id.get_exp_arr_in()
log_arr_in = rs2id.get_log_arr_in()

result = rs2id.rs2id(
    message, tag_pos, tag_pos_in, exp_arr, log_arr, exp_arr_in, log_arr_in, gf_exp
)
print(f"RS2ID result: {result}")
