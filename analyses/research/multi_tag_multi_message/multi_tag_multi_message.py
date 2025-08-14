import matplotlib.pyplot as plt
import numpy as np
import os

from idsys import IdMetrics, create_id_system

def theoretical_fp_rate(k, t, p):
    """
    Calculate theoretical false positive rate for k-identification with t tags.
    
    Args:
        k: Number of validation messages
        t: Number of tags
        p: Base false positive rate for single tag and message
        
    Returns:
        Theoretical false positive rate
    """
    # For k-identification with t tags:
    # P_fp = 1 - (1 - p^t)^k
    return 1 - (1 - p**t)**k


def main():
    print("=" * 80)
    print("ANALYSIS OF K-IDENTIFICATION AND MULTIPLE TAGS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("analyses/multi_tag_multi_message", exist_ok=True)
    
    # Base parameters
    vec_len = 4
    num_messages = 10**6  # Use fewer messages for faster runs
    base_rate = 2**(-8)   # Theoretical base rate for GF(2^8)
    trials = 1  # Number of trials for empirical evaluation with small num_messages
    n_id = 2**(vec_len)  # Number of possible IDs for GF(2^8) with vec_len in bytes
    
    # K-identification analysis (single tag)
    #test_range = np.linspace(1, 1000, num=21, dtype=int)  # k values from 1 to 1000
    test_range = np.unique(np.logspace(0, 4, num=21, dtype=int))  # Logarithmic scale for better distribution
    
    system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
    
    empirical_rates = []
    theoretical_rates = []
    
    print("\nK-IDENTIFICATION ANALYSIS (SINGLE TAG)")
    print("-" * 50)
    print(f"{'k':>5} | {'Empirical':>10} | {'Theoretical':>10} | {'Theoretical2':>10} | {'Ratio':>10}")
    print("-" * 50)
    

    for k in test_range:
        results = IdMetrics.evaluate_system(
            system=system,
            vec_len=vec_len,
            num_messages=num_messages,
            num_validation_messages=k
        )
        
        empirical = results['false_positive_rate']
        theoretical = theoretical_fp_rate(k, 1, base_rate)
        
        empirical_rates.append(empirical)
        theoretical_rates.append(theoretical)

    
    # Plotting k-identification results
    plt.figure(figsize=(10, 6))
    plt.plot(test_range, empirical_rates, '-', label="Empirical", linewidth=1)
    plt.plot(test_range, theoretical_rates, '--', label="Theoretical (1-(1-p)^k)", linewidth=1)
    
    plt.title("False Positive Rate vs Number of Validation Messages", fontsize=14)
    plt.xlabel("Number of Validation Messages (k)", fontsize=12)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xscale('log', base=10)
    plt.yscale('log', base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("analyses/multi_tag_multi_message/k_identification_analysis.png", dpi=300)
    
    # Multiple tags analysis (single validation message)
    tag_positions = [
        [2],           # 1 tag
        [2, 3],        # 2 tags
        [2, 3, 4],     # 3 tags
        [2, 3, 4, 5],  # 4 tags
    ]
    
    empirical_multi_tag = []
    theoretical_multi_tag = []
    
    print("\nMULTIPLE TAGS ANALYSIS (SINGLE MESSAGE)")
    print("-" * 50)
    print(f"{'Tags':>5} | {'Empirical':>15} | {'Theoretical':>15} | {'Ratio':>10}")
    print("-" * 50)
    
    for tag_pos in tag_positions:
        system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": tag_pos})
        
        results = IdMetrics.evaluate_system(
            system=system,
            vec_len=vec_len,
            num_messages=num_messages,
            num_validation_messages=1
        )
        
        num_tags = len(tag_pos)
        empirical = results['false_positive_rate']
        theoretical = base_rate**num_tags  # p^t
        
        empirical_multi_tag.append(empirical)
        theoretical_multi_tag.append(theoretical)
        
        ratio = empirical / theoretical if theoretical > 0 else 0
        print(f"{num_tags:5d} | {empirical:15.8e} | {theoretical:15.8e} | {ratio:10.4f}")
    
    # Combined analysis: Multiple tags with k-identification
    print("\nCOMBINED ANALYSIS: MULTIPLE TAGS WITH K-IDENTIFICATION")
    print("-" * 80)
    print(f"{'Tags':>5} | {'k':>5} | {'Empirical':>15} | {'Theoretical':>15} | {'Ratio':>10}")
    print("-" * 80)
    
    combined_results = []
    num_messages = 10**6
    ks = [100, 500, 1000]  # k values for combined analysis
    
    for tag_pos in [[2], [2, 3], [2, 3, 4]]:
        system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": tag_pos})
        num_tags = len(tag_pos)        
        empirical = []
        theoretical = []
        
        for k in ks:
            results = IdMetrics.evaluate_system(
                system=system,
                vec_len=vec_len,
                num_messages=num_messages,
                num_validation_messages=k
            )
            
            empirical.append(results['false_positive_rate'])
            theoretical.append(theoretical_fp_rate(k, num_tags, base_rate))            
            
        combined_results.append({
            'tags': num_tags,
            'k': ks,
            'empirical': empirical,
            'theoretical': theoretical
        })
        
    # plot fp rate vs k for multiple tags    
    for result in combined_results:
        plt.figure(figsize=(10, 6))
        print(f"Tags: {result['tags']}, k values: {result['k']}")
        print(f"Empirical: {result['empirical']}")
        print(f"Theoretical: {result['theoretical']}")
        print("-" * 80)
        plt.plot(result['k'], result['empirical'], 'o-', label=f"Tags: {result['tags']}", linewidth=1)
        plt.plot(result['k'], result['theoretical'], '--', linewidth=1)
        plt.title("False Positive Rate vs Number of Validation Messages (Multiple Tags)", fontsize=14)
        plt.xlabel("Number of Validation Messages (k)", fontsize=12)
        plt.ylabel("False Positive Rate", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"analyses/multi_tag_multi_message/combined_analysis_{result['tags']}.png", dpi=300)
    plt.show()

    print("\nAnalysis complete! Results saved in analyses/multi_tag_multi_message/")

if __name__ == "__main__":
    main()