import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from idsys import create_id_system, IdMetrics

def main():
    vec_len = 16
    gf_exp = 16  # 2**16 = 65536
    num_messages = 10**5

    # Create RSID system with tag at position 4
    rsid = create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]})

    # Evaluate system with incremental pattern
    result = IdMetrics.evaluate_system(
        system=rsid,
        message_pattern="incremental",
        vec_len=vec_len,
        num_messages=num_messages
    )

    tag_pdf = result['tag_pdf']
    all_symbols = np.arange(2**16)
    tag_probs = np.array([tag_pdf.get(symbol, 0.0) for symbol in all_symbols])

    plt.figure(figsize=(18, 5))
    plt.plot(all_symbols, tag_probs, marker='.', linestyle='-', linewidth=0.7, markersize=1, color='#E74C3C')
    plt.title("RSID Tag PDF (GF $2^{16}$) with Incremental Pattern")
    plt.xlabel("Tag Value")
    plt.ylabel("Probability")
    plt.xlim(0, 2**16 - 1)
    plt.tight_layout()
    plt.savefig("analyses/main/message_patterns/output/rsid_tag_pdf_gf16_incremental.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    main()