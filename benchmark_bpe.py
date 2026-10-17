from time import perf_counter

from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

input_path = FIXTURES_PATH / "corpus.en"

start_time = perf_counter()
run_train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)
elapsed = perf_counter() - start_time

print(f"BPE training time: {elapsed:.4f} seconds")