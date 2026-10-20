"""Benchmark the existing trainer and tokenizer on a bounded TinyStories subset.

Run from the project root with: uv run benchmark_tinystories.py
Cold encoding starts with a fresh tokenizer; warm encoding repeats the same text.
"""

import argparse
import codecs
import gc
import math
import resource
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from cs336_basics.tokenizer import BPETokenizer
from tests.adapters import run_train_bpe

MIB = 1024**2
SPECIAL_TOKEN = "<|endoftext|>"
ROOT = Path(__file__).resolve().parent


def read_prefix(path: Path, limit_mib: float) -> str:
    """Read a bounded prefix, preferring a complete document when truncating."""
    limit = int(limit_mib * MIB)
    with path.open("rb") as source:
        data = source.read(limit + 1)
    truncated = len(data) > limit
    data = data[:limit]
    if truncated:
        marker = SPECIAL_TOKEN.encode("utf-8")
        boundary = data.rfind(marker)
        if boundary >= 0:
            data = data[: boundary + len(marker)]
    decoder = codecs.getincrementaldecoder("utf-8")()
    # On a truncated prefix, hold back an incomplete final UTF-8 character.
    return decoder.decode(data, final=not truncated)


def peak_rss_mib() -> float:
    """Process lifetime peak RSS: bytes on macOS, KiB on Linux."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum / (MIB if sys.platform == "darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "tests/fixtures/tinystories_sample_5M.txt",
        help="TinyStories source file; only a bounded prefix is read.",
    )
    parser.add_argument("--train-mib", type=float, default=5, help="Maximum training subset size in MiB (default: 5).")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Target vocabulary size (default: 10000).")
    parser.add_argument(
        "--encode-mib",
        type=float,
        default=1,
        help="Maximum encoding sample size from the same source, in MiB; 0 skips encoding (default: 1).",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if not math.isfinite(args.train_mib) or int(args.train_mib * MIB) < 1:
        parser.error("--train-mib must specify at least one byte.")
    if not math.isfinite(args.encode_mib) or args.encode_mib < 0:
        parser.error("--encode-mib must be nonnegative.")
    if 0 < args.encode_mib * MIB < 1:
        parser.error("A nonzero --encode-mib must specify at least one byte.")
    if args.vocab_size < 257:
        parser.error("--vocab-size must be at least 257 (256 bytes plus the special token).")

    print(f"Source: {args.input.resolve()}", flush=True)
    source_mib = args.input.stat().st_size / MIB
    if args.train_mib > source_mib:
        print(f"Source contains only {source_mib:.3f} MiB; the training subset cannot be larger.", flush=True)

    training_text = read_prefix(args.input, args.train_mib)
    if not training_text:
        parser.error("The selected training prefix contains no complete UTF-8 characters.")
    with TemporaryDirectory(prefix="cs336-tinystories-benchmark-") as directory:
        subset_path = Path(directory) / "training_subset.txt"
        training_bytes = subset_path.write_bytes(training_text.encode("utf-8"))
        del training_text
        gc.collect()
        print(
            f"Training on {training_bytes / MIB:.3f} MiB with target vocabulary {args.vocab_size:,}...",
            flush=True,
        )
        started = perf_counter()
        vocab, merges = run_train_bpe(
            input_path=subset_path,
            vocab_size=args.vocab_size,
            special_tokens=[SPECIAL_TOKEN],
        )
        training_seconds = perf_counter() - started
        training_peak = peak_rss_mib()

    print(f"Training elapsed: {training_seconds:.3f} s", flush=True)
    print(f"Training process peak RSS: {training_peak:.1f} MiB (includes imports and subset preparation)", flush=True)
    print(f"Actual vocabulary: {len(vocab):,}; merges: {len(merges):,}", flush=True)

    if args.encode_mib == 0:
        return
    sample = read_prefix(args.input, args.encode_mib)
    sample_bytes = len(sample.encode("utf-8"))
    if sample_bytes == 0:
        parser.error("The selected encoding prefix contains no complete UTF-8 characters.")
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=[SPECIAL_TOKEN])
    print(f"Encoding {sample_bytes / MIB:.3f} MiB from the same source...", flush=True)
    cold_ids = None
    for label in ("Cold cache", "Warm cache"):
        started = perf_counter()
        ids = tokenizer.encode(sample)
        seconds = perf_counter() - started
        print(
            f"{label}: {seconds:.3f} s; {sample_bytes / MIB / seconds:.3f} MiB/s; "
            f"{len(ids) / seconds:,.0f} tokens/s; {len(ids):,} tokens",
            flush=True,
        )
        if cold_ids is None:
            cold_ids = ids
        elif ids != cold_ids:
            raise RuntimeError("Cold and warm encoding produced different token IDs.")
    print("Cold/warm token IDs match.", flush=True)


if __name__ == "__main__":
    main()
