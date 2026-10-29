"""Input/output path checks shared by the inference entry points."""

from pathlib import Path


def ensure_output_not_input(output, inputs):
    """Check every input before opening output, including links to the same file."""
    output = Path(output)
    resolved_output = output.resolve()
    for source in inputs:
        source = Path(source)
        if source.resolve() == resolved_output or (
            source.exists() and output.exists() and source.samefile(output)
        ):
            raise ValueError(f"Input and output must be different files: {source}")
