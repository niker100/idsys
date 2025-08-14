import os
import matplotlib.pyplot as plt
import json

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - NUMBER OF MESSAGES INFLUENCE")
    print("=" * 50)

    # Read system results from the file
    input_file = 'analyses/num_messages_influence/system_results.json'

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
    gf_exp = results_with_metadata['gf_exp']



    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot false positives vs number of messages
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['false_positives'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positives vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('False Positives', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)

    # Add textbox with trials, gf_exp, and vec_len
    textstr = f'GF Exp: {gf_exp}\nVec Len: {vec_len}'
    plt.gcf().text(0.8, 0.9, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/false_positives_vs_num_messages.png', dpi=300, bbox_inches='tight')

    # Plot execution time vs number of messages
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['exec_time'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Execution Time vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('Avg Execution Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)

    # Add textbox with trials, gf_exp, and vec_len
    textstr = f'GF Exp: {gf_exp}\nVec Len: {vec_len}'
    plt.gcf().text(0.8, 0.9, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/exec_time_vs_num_messages.png', dpi=300, bbox_inches='tight')

    # Plot false positive rate vs number of messages
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['fp_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positive Rate vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('False Positive Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)

    # Add textbox with trials, gf_exp, and vec_len
    textstr = f'GF Exp: {gf_exp}\nVec Len: {vec_len}'
    plt.gcf().text(0.8, 0.9, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/fp_rate_vs_num_messages.png', dpi=300, bbox_inches='tight')

   
if __name__ == "__main__":
    main()
