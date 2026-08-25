#!/usr/bin/env python3
"""Console-script entry point for q4nx-build."""
import os
import sys

from q4nx import create_converter, create_hf_converter
from q4nx.arch_detect import family_from_text
from q4nx.build_plan import (
    derive_build_plan,
    derive_build_plan_from_card,
    find_skeleton_for_arch,
    format_chain,
)
from q4nx.model_assets import (
    assemble_model_assets,
    assemble_model_assets_hf,
    get_default_flm_version,
    find_repo_gguf,
    select_repo_gguf,
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


def _is_card_source(path: str) -> bool:
    """True for a model-card-only input: a README.md file or a dir holding one.

    Dirs that carry actual weights/configs are real local sources, not cards.
    """
    if os.path.isfile(path):
        return path.lower().endswith((".md", ".markdown"))
    if os.path.isdir(path):
        if os.path.basename(path).lower() == "readme.md":
            return True
        has_card = os.path.isfile(os.path.join(path, "README.md"))
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
            or any(f.lower().endswith(".gguf") for f in os.listdir(path))
        )
        return has_card and not has_weights
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
        "-i", "--input", dest="input_flag",
        help="Input GGUF file, an HF repo id, or a model card (dir with "
             "README.md / a .md file) naming one",
    )
    parser.add_argument(
        "-o", "--output", dest="output_flag", help="Output folder (optional)"
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="weights_type",
        default=None,
        choices=["language", "vision", "audio"],
        help="Type of weights to convert (default: inferred from the repo card; "
             "VLM pipeline tags convert language + vision, otherwise language)",
    )
    parser.add_argument(
        "-f", "--force", dest="force_model_type", default="", help="Model type override"
    )
    parser.add_argument(
        "-s", "--source-model", dest="source_model", default=None,
        help="Source HF/ModelScope model for tokenizer/config assets (the NPU2 "
             "skeleton). Default: the first ancestor in the repo card's "
             "base_model chain with an {org}/{base}-NPU2 mirror "
             "(orgs: Atomic-Germ, then FastFlowLM).",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Resolve and print the build plan (GGUF choice, base_model chain, "
             "skeleton source, output name, weights type) without converting.",
    )
    parser.add_argument(
        "--pad-to-fit", dest="pad_to_fit", action="store_true",
        help="When the model's hidden size is smaller than the selected engine "
             "variant's official dim, zero-pad the hidden axis so the weights "
             "fit the compiled variant (padded channels are inert).",
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

    # Local paths must exist; HF repo ids are resolved later by the converter.
    if not _is_hf_repo_id(input_path) and not os.path.exists(input_path):
        sys.exit(f"Error: Input file does not exist: {input_path}")

    flm_version = args.flm_version or get_default_flm_version()

    # Fill unspecified -s/-o/-t from the repo card's base_model chain
    # (q4nx.build_plan): walk ancestors until one has an {org}/{base}-NPU2
    # mirror, use it as the skeleton source, and read the output name and VLM
    # detection from the same cards. Explicit flags always win.
    weights_type = args.weights_type
    source_model = args.source_model
    output_folder = args.output_flag
    family_hint = None
    plan = None
    if _is_card_source(input_path):
        print(f"[INFO] Reading model card: {input_path}")
        try:
            plan = derive_build_plan_from_card(input_path)
        except FileNotFoundError as e:
            sys.exit(f"Error: {e}")
        if not plan.repo_id:
            sys.exit("Error: Could not determine an HF repo id from this model card "
                     "(no org/name mention found in the body or frontmatter).")
        print(f"[INFO] Card names upstream repo: {plan.repo_id}")
        input_path = plan.repo_id
    if _is_hf_repo_id(input_path):
        if plan is None:
            plan = derive_build_plan(input_path)
        # Chain dead-end rescue: a repo may declare no base_model anywhere,
        # yet its config.json still reveals family+size, which maps to the
        # same {org}/{base}-NPU2 mirror convention (GRaPE-style cards).
        if plan.skeleton is None:
            try:
                from q4nx.model_converter import _detect_hf_arch

                arch = _detect_hf_arch(input_path)
            except Exception:
                arch = None
            if arch is not None:
                rescued = find_skeleton_for_arch(arch)
                if rescued:
                    plan.skeleton = rescued
                    print(f"[INFO] No base_model declared; config.json says "
                          f"{arch.name}, so using skeleton: {rescued}")
        print(f"[INFO] Base chain: {format_chain(plan.chain)}")
        if source_model is None:
            if plan.skeleton:
                print(f"[INFO] Skeleton source: {plan.skeleton}")
            else:
                print("[WARN] No {org}/{base}-NPU2 skeleton found on the chain; "
                      "pass -s to name the asset source explicitly.")
            source_model = plan.skeleton
        if weights_type is None:
            weights_type = plan.weights_type
            suffix = f" ({plan.weights_reason})" if plan.weights_reason else ""
            print(f"[INFO] Weights type: {weights_type}{suffix}")
        if output_folder is None and plan.output_name:
            output_folder = plan.output_name
            print(f"[INFO] Output folder: {output_folder} (from card metadata)")
        family_hint = family_from_text(" ".join(plan.chain))

    weights_type = weights_type or "language"
    output_folder = output_folder or os.path.dirname(input_path) or "."

    # Resolve the weight source before touching absolute paths: an HF repo id
    # must stay in 'org/name' form or _is_hf_repo_id/create_hf_converter won't
    # recognize it.
    #
    # -i <hf-repo-id> prefers a quantized GGUF shipped in the repo itself,
    # chosen in a family-preferred order (default q4_1, then q4_0, then q8_0).
    # The order is driven by -f when given, then by the chain-derived family
    # hint. The chosen GGUF is downloaded via the HF cache; if the repo has
    # none, we fall back to the HF-safetensors source path below.
    hf_input = None
    source_file = None
    selected_gguf = None
    if _is_hf_repo_id(input_path):
        if args.dry_run:
            selected_gguf = select_repo_gguf(input_path, args.force_model_type, family_hint)
            if selected_gguf is None:
                hf_input = input_path
        else:
            found = find_repo_gguf(input_path, args.force_model_type, family_hint=family_hint)
            if found is not None:
                input_path, source_file = found
                source_model = source_model or input_path
            else:
                hf_input = input_path
    elif _is_hf_source(input_path):
        hf_input = input_path

    if args.dry_run:
        requested = args.input_flag or args.input_file
        print("[DRY-RUN] Build plan (nothing was converted or downloaded):")
        print(f"  input:         {requested}")
        if selected_gguf:
            print(f"  source GGUF:   {selected_gguf}")
        elif hf_input:
            print("  source:        HF safetensors (no quantized GGUF in the repo)")
        elif os.path.exists(requested):
            print(f"  source GGUF:   {requested}")
        print(f"  skeleton (-s): {source_model or '(GGUF provenance fallback)'}")
        print(f"  weights (-t):  {weights_type}")
        print(f"  output (-o):   {os.path.abspath(output_folder)}")
        return 0

    output_folder = os.path.abspath(output_folder)
    os.makedirs(os.path.dirname(output_folder) or ".", exist_ok=True)

    print(f"[INFO] Converting {input_path} to {output_folder}...")

    if hf_input is not None:
        model = create_hf_converter(hf_input, args.force_model_type)
        model.pad_to_fit = args.pad_to_fit
        if weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=weights_type)
        assemble_model_assets_hf(
            model.hf_source,
            model.q4nx_config,
            output_folder,
            source_model=source_model or hf_input,
            flm_version=flm_version,
            source_file=source_file,
            model_arch=model.model_arch,
        )
    else:
        model = create_converter(input_path, args.force_model_type)
        model.pad_to_fit = args.pad_to_fit
        if weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=weights_type)
        assemble_model_assets(
            model.gguf_reader,
            model.q4nx_config,
            output_folder,
            source_model=source_model,
            flm_version=flm_version,
            source_file=source_file,
            model_arch=model.model_arch,
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