"""
wget https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_automatic/saspeech_automatic.7z
wget https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_manual/saspeech_manual_v1.7z
sudo apt install p7zip-full -y
7z x saspeech_automatic.7z
7z x saspeech_manual_v1.7z

uv run src/prepare.py --input_folder saspeech_automatic saspeech_manual --output_folder dataset
rm -r saspeech_automatic saspeech_manual
"""

import argparse
import os
from pathlib import Path
from config import SEP

SRC_SEP = "\t"

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, nargs="+", required=True)
parser.add_argument("--output_folder", type=str, required=True)
args = parser.parse_args()

def main():
    output_path = Path(args.output_folder)
    wav_output_path = output_path / "wav"
    wav_output_path.mkdir(parents=True, exist_ok=True)
    
    combined_data = []
    counter = 0
    
    for input_folder in args.input_folder:
        input_path = Path(input_folder)
        metadata_path = input_path / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.csv in {input_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split(SRC_SEP)
                if len(parts) < 2:
                    continue

                index = parts[0]
                text = parts[1].strip()

                original_wav = input_path / "wav" / f"{index}.wav"
                new_wav = wav_output_path / f"{counter}.wav"

                if original_wav.exists():
                    os.link(original_wav, new_wav)
                    combined_data.append(f"{counter}{SEP}{text}")
                    counter += 1
    
    # Save combined metadata
    with open(output_path / "metadata.csv", "w") as f:
        f.write("\n".join(combined_data))
    
    print(f"Total files processed: {len(combined_data)}")


if __name__ == "__main__":
    main()
