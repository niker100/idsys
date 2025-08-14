"""
Plot execution time vs vector length for multiple gf_exp values and systems.
Each plot shows all gf_exp curves for one system. Multiple plots for multiple systems.
"""

import matplotlib.pyplot as plt
import os
from idsys import IdMetrics, create_id_system
import tracemalloc

def main():
    print("=" * 60)
    print("EXECUTION TIME AND MEMORY USAGE VS VECTOR LENGTH")
    print("=" * 60)

    # Parameters
    vec_lengths = [2**i for i in range(4, 17, 3)]
    gf_exp_values = [8, 16, 32, 64]
    system_types = [
        ("RSID", lambda gf_exp: lambda vec_len: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [1]})),
        ("RMID", lambda gf_exp: lambda vec_len: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [1]})),
        ("RS2ID", lambda gf_exp: lambda vec_len: create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": 1, "tag_pos_in": 1})),
        ("SHA1ID", lambda gf_exp: lambda vec_len: create_id_system("SHA1ID", {"gf_exp": gf_exp})),
        ("SHA256ID", lambda gf_exp: lambda vec_len: create_id_system("SHA256ID", {"gf_exp": gf_exp})),
    ]
    
    num_messages = 100
    os.makedirs("analyses/framework_performance", exist_ok=True)
    
    # For each system, collect data
    for system_name, sys_factory in system_types:
        print(f"\nSystem: {system_name}")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        for gf_exp in gf_exp_values:
            if system_name == "RS2ID" and gf_exp > 32:
                continue
            if system_name == "RMID" and gf_exp > 32:
                continue
                
            exec_times = []
            memory_usages = []
            
            for vec_len in vec_lengths:
                print(f"  gf_exp={gf_exp}, vec_len={vec_len} ...", end="", flush=True)
                
                # Clear memory and start tracking
                tracemalloc.clear_traces()
                tracemalloc.start()
                
                # Create system and run evaluation
                system = sys_factory(gf_exp)(vec_len)
                metrics = IdMetrics.evaluate_system(
                    system=system,
                    vec_len=vec_len,
                    num_messages=num_messages,
                    num_processes=1,
                )
                
                # Get peak traced memory
                _, peak_traced = tracemalloc.get_traced_memory()
                traced_mb = peak_traced / (1024 * 1024)
                
                # Record metrics
                exec_times.append(metrics["avg_execution_time_ms"])
                memory_usages.append(traced_mb)
                
                print(f" {metrics['avg_execution_time_ms']:.3f} ms, {traced_mb:.1f} MB")
                
                # Clean up
                del system, metrics
                tracemalloc.stop()
            
            # Plot data
            ax1.plot(vec_lengths, exec_times, marker='o', label=f"GF_EXP={gf_exp}")
            ax2.plot(vec_lengths, memory_usages, marker='s', label=f"GF_EXP={gf_exp}")

        # Configure plots
        ax1.set_title(f"Execution Time vs Vector Length for {system_name}", fontsize=15)
        ax1.set_xlabel("Vector Length", fontsize=13)
        ax1.set_ylabel("Avg Execution Time (ms)", fontsize=13)
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title(f"Memory Usage vs Vector Length for {system_name}", fontsize=15)
        ax2.set_xlabel("Vector Length", fontsize=13)
        ax2.set_ylabel("Memory Usage (MB)", fontsize=13)
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"analyses/framework_performance/performance_{system_name}.png", dpi=300)
        print(f"Saved plot: analyses/framework_performance/performance_{system_name}.png")
        plt.close()


if __name__ == "__main__":
    main()