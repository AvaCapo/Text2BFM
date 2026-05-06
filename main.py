"""Simple project entrypoint.

Run with:
    python main.py

Any extra CLI arguments are forwarded to Hydra in `train_text2bfm.py`.
"""

from train_text2bfm import main as train_main


def main() -> None:
    """Start the default generator training pipeline."""
    train_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
