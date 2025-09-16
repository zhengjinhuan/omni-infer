import os
import sys
import time
import torch

def main(input_dir, output_dir):
    files = os.listdir(input_dir)
    data = []
    for file in files:
        filepath = os.path.join(input_dir, file)
        data.extend(torch.load(filepath))

    torch.save(data, os.path.join(output_dir, f"merged-tokens-{time.time_ns()}.pt"))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])