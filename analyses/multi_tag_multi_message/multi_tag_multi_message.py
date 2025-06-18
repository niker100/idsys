import matplotlib.pyplot as plt
import numpy as np
from framework import IdMetrics, create_id_system

def main():
    systems = [
        create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]}),
        create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2, 3]}),
        create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2, 3, 4]}),
    ]
               
    vec_len = 16
    num_messages = 100000
    nums_validation_messages = [10, 50, 100]

    # Multiple valid messages at receiver
    for num_validation_messages in nums_validation_messages:
        results = IdMetrics.evaluate_system(
            system=systems[0],
            vec_len=vec_len,
            num_messages=num_messages,
            num_validation_messages=num_validation_messages
        )
        print(f"fp_rate for num_validation_messages={num_validation_messages}: {results['false_positive_rate']:.6f}, theoretical: {num_validation_messages*2**(-8):.6f}")

    # Multiple tags sent
    for system in systems:
        results = IdMetrics.evaluate_system(
            system=system,
            vec_len=vec_len,
            num_messages=num_messages,
            num_validation_messages=1
        )
        num_tags = len(system.encoder.parameters.get('tag_pos', []))
        print(f"fp_rate for {num_tags} tags: {results['false_positive_rate']:.8f}, theoretical: {2**(-8*num_tags):.8f}")

    # Combination (theoretical values not verified)
    for num_validation_messages in nums_validation_messages:
        results = IdMetrics.evaluate_system(
            system=systems[1],
            vec_len=vec_len,
            num_messages=num_messages,
            num_validation_messages=num_validation_messages
        )
        num_tags = len(systems[1].encoder.parameters.get('tag_pos', []))
        print(f"fp_rate for {num_tags} tags and num_validation_messages={num_validation_messages}: {results['false_positive_rate']:.8f}, theoretical: {num_validation_messages*2**(-8*num_tags):.8f}")

if __name__ == "__main__":
    main()