"""
Check if Galois Field (GF) table generation is required or if on-the-fly calculations are used.

This script benchmarks GF table generation and encoding for various GF classes and exponents,
and reports whether the idcodes implementation uses precomputed tables or on-the-fly arithmetic.

For each GF class and exponent:
- Attempts to generate tables and catch MemoryError.
- Times encoding with and without explicit table generation.
- Reports memory usage and timing.
- Indicates if encoding works without precomputed tables (i.e., on-the-fly).

Author: Your Name
"""

import time
import psutil
from idcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64

def memory_usage_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def try_generate_tables(gf, gf_exp):
    # GF32 and GF64 will always fail or crash, so skip
    if gf_exp >= 32:
        print(f"  [SKIP] Skipping GF(2^{gf_exp}) table generation (would cause OOM or segfault).")
        return False
    try:
        start_mem = memory_usage_mb()
        gf.generate_gf_outer(gf_exp)
        outer_mem = memory_usage_mb()
        gf.generate_gf_inner(gf_exp)
        inner_mem = memory_usage_mb()
        print(f"  [OK] GF(2^{gf_exp}) tables generated. Δmem outer: {outer_mem-start_mem:.1f} MB, inner: {inner_mem-outer_mem:.1f} MB")
        return True
    except MemoryError:
        print(f"  [FAIL] MemoryError during GF(2^{gf_exp}) table generation.")
        return False
    except Exception as e:
        print(f"  [FAIL] Exception during GF(2^{gf_exp}) table generation: {e}")
        return False

def try_encode(gf, gf_exp, use_tables):
    try:
        msg = gf.generate_string_sequence(16)
        start = time.perf_counter()
        tag = gf.rsid(msg, 2, gf.get_exp_arr(), gf.get_log_arr(), gf_exp)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  [OK] RSID encoding succeeded in {elapsed:.2f} ms (tables={'yes' if use_tables else 'no'}), tag: {tag}")
        return True
    except Exception as e:
        print(f"  [FAIL] RSID encoding failed (tables={'yes' if use_tables else 'no'}): {e}")
        return False

def analyse_gf_class(gf_class, gf_exp):
    print(f"\n=== Analysing {gf_class.__name__} with GF(2^{gf_exp}) ===")
    gf = gf_class()
    tables_ok = try_generate_tables(gf, gf_exp)
    if tables_ok:
        try_encode(gf, gf_exp, use_tables=True)
    else:
        if gf_exp < 32:
            print("  Skipping encoding with tables due to failed table generation.")

    # Always try on-the-fly encoding (no tables)
    gf2 = gf_class()
    try_encode(gf2, gf_exp, use_tables=False)

def main():
    print("GF Table vs On-the-fly Calculation Analysis\n")
    configs = [
        (IDCODES_U8, 8),
        (IDCODES_U16, 16),
        (IDCODES_U32, 32),
        (IDCODES_U64, 64),
    ]
    for gf_class, gf_exp in configs:
        analyse_gf_class(gf_class, gf_exp)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()