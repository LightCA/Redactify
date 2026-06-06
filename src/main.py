import os
import sys

from pathlib import Path

from cli import RedactifyCLI
from redactify import Redactify

if __name__ == "__main__":
    args = RedactifyCLI().parse()

    input_path = args.input_path
    output_path = args.output_path

    if input_path is None:
        valid_extensions_str = [f"*{ext}" for ext in Redactify.valid_extensions]

        import crossfiledialog

        input_path = Path(
            crossfiledialog.open_file(
                title="Select video file to censor", filter={"Video files": valid_extensions_str, "All files": "*.*"}
            )
        )

    if output_path is None:
        working_dir = Path(os.environ["WORKING_DIRECTORY"])
        output_dir = working_dir / "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.mp4"

    kwargs = {k: v for k, v in vars(args).items() if k not in ("input_path", "output_path")}

    red = Redactify()
    red.run(input_path, output_path, **kwargs)
