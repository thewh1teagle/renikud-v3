import argparse
import datasets

datasets.config.AUDIO_DECODING_BACKEND = "soundfile"

SEP = "\t"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ivrit-ai/whisper-large-v3-turbo")
    parser.add_argument("--output_dir", type=str, default="./whisper-heb-nikud")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["wandb", "tensorboard"])
    parser.add_argument("--dataset_cache_path", type=str, default="./.dataset-cache")
    return parser.parse_args()
