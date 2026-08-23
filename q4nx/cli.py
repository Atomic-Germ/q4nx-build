#!/usr/bin/env python3
"""Console-script entry point for q4nx-build."""
import os
import sys

from q4nx import create_converter, create_hf_converter
from q4nx.model_assets import (
    assemble_model_assets,
    assemble_model_assets_hf,
    get_default_flm_version,
    find_repo_gguf,
)


def _is_hf_repo_id(path: str) -> bool:
    """True if path looks like an 'org/name' HF repo id (not a local path, not a .gguf)."""
    if not path or path.endswith(".gguf") or os.path.exists(path):
        return False
    if path.startswith(("http://", "https://", "file:")):
        return False
    parts = path.split("/")
    return len(parts) == 2 and all(parts) and "\\" not in path


def _is_hf_source(path: str) -> bool:
    """True if path is a local HF-safetensors model dir."""
    if os.path.isdir(path):
        return (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        )
    return False


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        prog="q4nx-build",
        description=(
            "Convert GGUF or HF-safetensors model files to Q4NX format (output always named "
            "model.q4nx). -i also accepts an HF repo id: a quantized GGUF is auto-selected in "
            "family-preferred order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", nargs="?", help="Input GGUF file (positional)")
    parser.add_argument(
        "-i", "--input", dest="input_flag", help="Input GGUF file, or an HF repo id"
    )
    parser.add_argument(
        "-o", "--output", dest="output_flag", help="Output folder (optional)"
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="weights_type",
        default="language",
        choices=["language", "vision", "audio"],
    )
    parser.add_argument(
        "-f", "--force", dest="force_model_type", default="", help="Model type override"
    )
    parser.add_argument(
        "-s", "--source-model", dest="source_model", default=None, help="Source HF/ModelScope model"
    )
    parser.add_argument(
        "--flm-version", dest="flm_version", default=None, help="flm_version to write into config.json"
    )
    parser.add_argument(
        "-d", "--deploy", dest="deploy_tag", default=None, metavar="NAME:SIZE",
        help="Deploy the converted model into flm's models dir and register it under this tag (e.g. 'qwen3.5-claude:9b')",
    )
    parser.add_argument(
        "--deploy-from", dest="deploy_from", default=None, metavar="SOURCE_TAG",
        help="Official registry entry to copy defaults from (e.g. 'qwen3.5:9b')",
    )
    parser.add_argument(
        "--deploy-name", dest="deploy_name", default=None, metavar="DIR",
        help="Directory name inside flm's models dir (default: derived from the deploy tag)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_path = args.input_flag or args.input_file
    if not input_path:
        sys.exit("Error: Input file is required. Use -i <file> or provide as positional argument.")

    output_folder = args.output_flag or os.path.dirname(input_path) or "."

    # Local paths must exist; HF repo ids are resolved later by the converter.
    if not _is_hf_repo_id(input_path) and not os.path.exists(input_path):
        sys.exit(f"Error: Input file does not exist: {input_path}")

    flm_version = args.flm_version or get_default_flm_version()

    # Resolve the weight source before touching absolute paths: an HF repo id
    # must stay in 'org/name' form or _is_hf_repo_id/create_hf_converter won't
    # recognize it.
    #
    # -i <hf-repo-id> prefers a quantized GGUF shipped in the repo itself,
    # chosen in a family-preferred order (default q4_1, then q4_0, then q8_0).
    # The order is driven by -f when given. The chosen GGUF is downloaded via
    # the HF cache; if the repo has none, we fall back to the HF-safetensors
    # source path below.
    hf_input = None
    source_file = None
    source_model = args.source_model
    if _is_hf_repo_id(input_path):
        found = find_repo_gguf(input_path, args.force_model_type)
        if found is not None:
            input_path, source_file = found
            source_model = source_model or input_path
        else:
            hf_input = input_path
    elif _is_hf_source(input_path):
        hf_input = input_path

    output_folder = os.path.abspath(output_folder)
    os.makedirs(os.path.dirname(output_folder) or ".", exist_ok=True)

    print(f"[INFO] Converting {input_path} to {output_folder}...")

    if hf_input is not None:
        model = create_hf_converter(hf_input, args.force_model_type)
        if args.weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=args.weights_type)
        assemble_model_assets_hf(
            model.hf_source,
            model.q4nx_config,
            output_folder,
            source_model=source_model or hf_input,
            flm_version=flm_version,
            source_file=source_file,
        )
    else:
        model = create_converter(input_path, args.force_model_type)
        if args.weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=args.weights_type)
        assemble_model_assets(
            model.gguf_reader,
            model.q4nx_config,
            output_folder,
            source_model=source_model,
            flm_version=flm_version,
            source_file=source_file,
        )

    if args.deploy_tag:
        from q4nx.deploy import deploy_model

        deploy_model(
            output_folder,
            args.deploy_tag,
            model.model_arch,
            model_dir_name=args.deploy_name,
            deploy_from=args.deploy_from,
        )

    print(f"[INFO] Conversion complete! Output saved to {output_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())