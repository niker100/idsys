"""Test for examining the tag size
"""
from idsys import generate_test_messages, create_id_system
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - TAG SIZE")
    print("=" * 50)

    vec_lengths = [4, 8, 16, 32, 64, 128]
    #vec_lengths = [10, 30, 50, 70, 90, 110]

    # Create systems as a dictionary for compare_systems
    systems = {
        "RSID": lambda vec_len: create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2, "vec_len": vec_len}),
        "RMID": lambda vec_len: create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2, "vec_len": vec_len}),
        "SHA1ID": lambda vec_len: create_id_system("SHA1ID", {"gf_exp": 8, "vec_len": vec_len})
    }

    # Store results for each system
    system_results = {name: {'vec_length': [], 'reliability': [], 'exec_time': [], 'code_rate': []} for name in systems.keys()}

    for vec_len in vec_lengths:

        # Generate test messages for this vec_length
        print(f"\n\033[4mEvaluating with vec_length: {vec_len}\033[0m")
        messages = generate_test_messages(vec_len=vec_len, gf_exp=8, count=1000
                                          )

        system_instances = {name: make_sys(vec_len) for name, make_sys in systems.items()}
        for name, system in system_instances.items():
            tag = []
            for message in messages:
                    tag.append(system.send(message))
            #print(tag)
            print(f"{name}: max tag: {max(tag)} => bits: {max(tag).bit_length()}")

    gf_exp_values = [8, 16]
    
    systems = {
        "RSID": lambda gf_exp: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": 2, vec_len: 16}),
        "RMID": lambda gf_exp: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": 2, vec_len: 16}),
        "SHA1ID": lambda gf_exp: create_id_system("SHA1ID", {"gf_exp": gf_exp, "vec_len": 16})
    }

    for gf_exp in gf_exp_values:
        print(f"\n\033[4mEvaluating with GF_EXP: {gf_exp}\033[0m")
        # Generate test messages for this gf_exp
        messages = generate_test_messages(vec_len=16, gf_exp=gf_exp, count=1000)
        system_instances = {name: make_sys(gf_exp) for name, make_sys in systems.items()}
        for name, system in system_instances.items():
            tag = []
            for message in messages:
                tag.append(system.send(message))
            #print(tag)
            print(f"{name} max tag: {max(tag)} => bits: {max(tag).bit_length()}")  
            
    print("\nTAG SIZE FINDINGS:")
    print("-" * 30)
    print("tag size is proportional to the gf_exp\n\n")
    

if __name__ == "__main__":
    main()