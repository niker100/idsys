import numpy as np
from ecidcodes.idcodes_gf import GF_U8

# Create an instance of the GF class for uint8_t
gf = GF_U8()

gf_exp = 8

gf.generate_gf_outer(8)
gf.generate_gf_inner(8)
print(gf.get_exp_arr())
print(gf.get_log_arr())

exp_arr_in = gf.get_exp_arr_in()
log_arr_in = gf.get_log_arr_in()
# Call the GF multiplication function
print(exp_arr_in)
print(log_arr_in)
