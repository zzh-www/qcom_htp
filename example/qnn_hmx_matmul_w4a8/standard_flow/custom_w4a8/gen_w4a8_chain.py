#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "qnn_hmx_matmul_common")))

from gen_quant_chain import main


if __name__ == "__main__":
    main("w4a8")
