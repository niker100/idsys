import numpy as np
from ecidcodes.idcodes_rmid import RMID_U8

# Create an instance of the RMID class for uint8_t
rmid = RMID_U8()

# Define parameters
message = [1, 2, 3, 4]
tag_pos = 5
rm_order = 2
gf_exp = 8

rmid.generate_gf_outer(gf_exp)

exp_arr = rmid.get_exp_arr()
log_arr = rmid.get_log_arr()


result = rmid.rmid(message, tag_pos, rm_order, exp_arr, log_arr, gf_exp)
print(f"RMID result: {result}")


eval_point_rm = [1, 2, 3, 4]
k_rm = 3
monomials = rmid.generate_monomials(
    rm_order, eval_point_rm, k_rm, exp_arr, log_arr, gf_exp
)
print(f"Generated monomials: {monomials}")
