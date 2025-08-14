import os
import matplotlib.pyplot as plt
import json

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - GF_EXP INFLUENCE")
    print("=" * 50)

    # Read system results from the file
    input_file = 'analyses/gf_exp_influence/system_results.json'

    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            results_with_metadata = json.load(f)

        print("System results loaded from", input_file)
        #print(json.dumps(results_with_metadata, indent=4))  # Print the loaded data for verification
    else:
        print(f"File {input_file} does not exist.")

    vec_len = results_with_metadata['vec_len']
    num_messages = results_with_metadata['num_messages']
    system_results = results_with_metadata['system_results']
    gf_exp_values = results_with_metadata['gf_exp_values']

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot execution time vs gf_exp
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['exec_time'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Execution Time vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)

    # Add textbox with trials, vec_len, and num_messages
    textstr = f'Vec Len: {vec_len}\nNum Messages: {num_messages}'
    plt.gcf().text(0.87, 0.9, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/exec_time_vs_gf_exp.png', dpi=300, bbox_inches='tight')

    # Plot false positives vs GF_EXP
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['false_positives'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positives vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('False Positives', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)

    # Add textbox with trials, vec_len, and num_messages
    textstr = f'Vec Len: {vec_len}\nNum Messages: {num_messages}'
    plt.gcf().text(0.87, 0.9, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/false_positives_vs_gf_exp.png', dpi=300, bbox_inches='tight')


    # Plot false positive rate vs gf_exp
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['fp_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positive Rate vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('False Positive Rate', fontsize=12)
    plt.grid(True, alpha=0.3)    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)

    # Add textbox with trials, vec_len, and num_messages
    textstr = f'Vec Len: {vec_len}\nNum Messages: {num_messages}'
    plt.gcf().text(0.87, 0.9, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/fp_rate_vs_gf_exp.png', dpi=300, bbox_inches='tight')


    # Print all false positives for each system
    print("\n" + "." * 50)
    print("FALSE POSITIVES")
    print("." * 50)
    for system_name, results in system_results.items():
        print(f"{system_name}:")
        for gf_exp, false_positives in zip(results['gf_exp'], results['false_positives']):
            print(f"  GF_EXP {gf_exp}: False Positives = {false_positives}")

    return

if __name__ == "__main__":
    main()


'''
RSID:
  GF_EXP 8: False Positives = 992
  GF_EXP 16: False Positives = 9
  GF_EXP 32: False Positives = 0
  GF_EXP 64: False Positives = 0
RMID:
  GF_EXP 8: False Positives = 1068
  GF_EXP 16: False Positives = 12
  GF_EXP 32: False Positives = 0
  GF_EXP 64: False Positives = 0
SHA1ID:
  GF_EXP 8: False Positives = 1028
  GF_EXP 16: False Positives = 5
  GF_EXP 32: False Positives = 0
  GF_EXP 64: False Positives = 0
'''