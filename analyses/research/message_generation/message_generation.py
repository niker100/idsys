"""Test for examining the influence of the message generation
"""
from idsys import generate_test_messages
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - Examination of message generation")
    print("=" * 50)

    for vl in range(1,10):
        np.random.seed(2)
        messages = generate_test_messages(vec_len = vl, gf_exp= 9, count=1)
        #print(type(messages[0][0]))
        print(messages)


if __name__ == "__main__":
    main()
