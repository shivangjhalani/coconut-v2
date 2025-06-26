import argparse
import os
from typing import Optional
import base64

import pandas as pd  # type: ignore


def parquet_to_json(parquet_path: str, output_path: Optional[str] = None):
    """Convert a ScienceQA parquet file to a JSON-Lines file.

    Each row of the parquet table is serialized as one JSON object per line, so
    the semantics of the original dataframe are preserved *as is*.
    """
    # Read parquet with pandas (pyarrow engine)
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Re-encode any binary cells (e.g., PNG bytes) to base64 strings so that
    # pandas → json conversion does not choke on non-UTF8 data.

    def _encode_bytes(x):
        if isinstance(x, (bytes, bytearray)):
            return base64.b64encode(x).decode("ascii")
        return x

    df = df.applymap(_encode_bytes)

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        output_path = os.path.join(os.path.dirname(parquet_path), f"{base_name}.jsonl")

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write out – one json object per line
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"[scienceqa_to_json] Wrote {len(df)} records → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ScienceQA parquet to JSON Lines and save next to the source file."
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to the ScienceQA parquet file (e.g. data/train-00000-of-00001-...parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional custom path for the JSONL output (default: <same_dir>/<same_name>.jsonl)",
    )

    args = parser.parse_args()
    parquet_to_json(args.parquet_path, args.output)