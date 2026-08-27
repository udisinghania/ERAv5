"""Entry point for the Session 9-aligned 17M RMSNorm/SwiGLU MTP model."""

from pathlib import Path
import sys

from train_mtp_98m import main


if __name__ == "__main__":
    if not any(
        argument == "--config" or argument.startswith("--config=")
        for argument in sys.argv[1:]
    ):
        sys.argv.extend(
            [
                "--config",
                str(Path(__file__).with_name("mtp_17m_session_config.json")),
            ]
        )
    main()
