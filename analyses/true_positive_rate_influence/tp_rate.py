"""Test for examining the influence of the true positive rate on the reliability of the identification
"""
from framework import IdMetrics
from framework import create_id_system, generate_test_messages, RSIDEncoder
import matplotlib.pyplot as plt

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - TRUE POSITIVE RATE INFLUENCE")
    print("=" * 50)

    # Create RSID system
    rsid_system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2})

    # Generate test messages
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=100)

    p_true_positives = [i / 100 for i in range(0, 101, 5)]  # True positive rates from 0.0 to 1.0
    results = []

    for tp_rate in p_true_positives:
        print(f"\nEvaluating with True Positive Rate: {tp_rate}")

        # Evaluate the system with generated messages
        metrics = IdMetrics.evaluate_system(
            rsid_system,
            messages,
            num_trials=100000,
            timing_iterations=0,
            p_true_positive=tp_rate,
        )

        results.append((tp_rate, metrics["reliability"], metrics["false_positive_rate"]))

    # Small visualization with matplotlib    
    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in results], [x[1] for x in results], marker='o')
    plt.title('Reliability vs True Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.ylabel('Reliability')
    plt.xticks(p_true_positives)
    plt.grid()
    plt.savefig('analyses/true_positive_rate_influence/reliability_influence.png')

    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in results], [x[2] for x in results], marker='o', color='orange')
    plt.title('False Positive Rate vs True Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.xticks(p_true_positives)
    plt.grid()
    plt.savefig('analyses/true_positive_rate_influence/fpr_influence.png')

if __name__ == "__main__":
    main()