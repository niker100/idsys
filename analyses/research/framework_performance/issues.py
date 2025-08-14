"""
Demonstrate issues with RMID and RS2ID usage in ecidcodes library.

- RMID: Throws errors when trying to encode.
- RS2ID: Not clear how to setup LUTs (inner/outer GF tables), only one initialize function, segmentation faults possible.
"""

from ecidcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64

def demo_rmid_problem():
    print("=== RMID Problem Demo ===")
    try:        
        message = [1, 2, 3, 4, 5, 6, 7, 8]
        tag_pos = 2
        rm_order = 1

        # This works
        gf_exp = 32
        idcodes = IDCODES_U32()
        tag = idcodes.rmid(message, tag_pos, rm_order, gf_exp)
        print("RMID tag:", tag)
        
        # This call throws error: RMID encoding failed: IndexError('tag_pos exceeds codeword positions in the chosen GF.')
        gf_exp = 64
        idcodes = IDCODES_U64()
        tag = idcodes.rmid(message, tag_pos, rm_order, gf_exp)
        print("RMID tag:", tag)
    except Exception as e:
        print("RMID encoding failed:", repr(e))

def demo_rs2id_problem():
    print("\n=== RS2ID Problem Demo ===")
    try:        
        message = [1, 2, 3, 4, 5, 6, 7, 8]
        tag_pos = 2
        tag_pos_in = 2

        # For gf_exp = 8, this works
        idcodes = IDCODES_U16()
        gf_exp = 8
        # RS2ID requires both inner and outer GF tables, but only one initialize_gf function is available.
        idcodes.generate_gf_outer(gf_exp * 2)
        idcodes.generate_gf_inner(gf_exp)
        # Only one initialize_gf function, not clear which tables to use.
        # The following line may cause a segmentation fault or undefined behavior:
        idcodes.initialize_gf(idcodes.get_exp_arr(), idcodes.get_log_arr(), gf_exp * 2)
        tag = idcodes.rs2id(message, tag_pos, tag_pos_in, gf_exp * 2)        
        print("RS2ID tag:", tag)

        # For gf_exp = 16, you get a segfault/kill when trying to initialize_gf
        idcodes = IDCODES_U32()
        gf_exp = 16
        idcodes.generate_gf_outer(gf_exp * 2)
        idcodes.generate_gf_inner(gf_exp)

        idcodes.initialize_gf(idcodes.get_exp_arr(), idcodes.get_log_arr(), gf_exp * 2)
        print("RS2ID tag:", tag)

    except Exception as e:
        print("RS2ID encoding failed:", repr(e))

if __name__ == "__main__":
    demo_rmid_problem()
    demo_rs2id_problem()

# OUTPUT:
# === RMID Problem Demo ===
# RMID tag: 1
# RMID encoding failed: IndexError('tag_pos exceeds codeword positions in the chosen GF.')

# === RS2ID Problem Demo ===
# RS2ID tag: 25709
# Killed