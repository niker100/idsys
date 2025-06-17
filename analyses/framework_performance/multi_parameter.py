"""
Plot execution time vs vector length for multiple gf_exp values and systems.
Each plot shows all gf_exp curves for one system. Multiple plots for multiple systems.
"""

import matplotlib.pyplot as plt
import numpy as np
from framework import IdMetrics, create_id_system, generate_test_messages

def main():
    print("=" * 60)
    print("EXECUTION TIME VS VECTOR LENGTH FOR MULTIPLE GF_EXP AND SYSTEMS")
    print("=" * 60)

    # Parameters
    vec_lengths = [2**i for i in range(1, 16)]
    gf_exp_values = [8, 16, 32, 64]
    system_types = [
        ("RSID", lambda gf_exp: lambda vec_len: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": 2})),
        ("RMID", lambda gf_exp: lambda vec_len: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": 2})),
        ("RS2ID", lambda gf_exp: lambda vec_len: create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": 2, "tag_pos_in": 2})),
        ("SHA1ID", lambda gf_exp: lambda vec_len: create_id_system("SHA1ID", {"gf_exp": gf_exp})),
        ("SHA256ID", lambda gf_exp: lambda vec_len: create_id_system("SHA256ID", {"gf_exp": gf_exp})),
    ]

    num_messages = 5000
    # For each system, collect execution time for all gf_exp and vec_lengths
    for system_name, sys_factory in system_types:
        print(f"\nSystem: {system_name}")
        plt.figure(figsize=(12, 7))
        for gf_exp in gf_exp_values:
            exec_times = []
            for vec_len in vec_lengths:
                print(f"  gf_exp={gf_exp}, vec_len={vec_len} ...", end="", flush=True)
                # messages = generate_test_messages(vec_len=vec_len, gf_exp=gf_exp, count=100)
                system = sys_factory(gf_exp)(vec_len)
                metrics = IdMetrics.evaluate_system(
                    system=system,
                    vec_len=vec_len,
                    num_messages=num_messages,
                    message_subset_size=10
                )
                exec_times.append(metrics["avg_execution_time_ms"])
                print(f" {metrics['avg_execution_time_ms']:.3f} ms")
            plt.plot(vec_lengths, exec_times, marker='o', label=f"GF_EXP={gf_exp}")

        plt.title(f"Execution Time vs Vector Length for {system_name} - {num_messages} Messages", fontsize=15, fontweight='bold')
        plt.xlabel("Vector Length", fontsize=13)
        plt.ylabel("Avg Execution Time (ms)", fontsize=13)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"analyses/framework_performance/exec_time_vs_vec_length_{system_name}.png", dpi=300, bbox_inches='tight')
        print(f"Saved plot: analyses/framework_performance/exec_time_vs_vec_length_{system_name}.png")

if __name__ == "__main__":
    main()