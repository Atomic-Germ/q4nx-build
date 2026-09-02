"""Derive build parameters (-s skeleton, -o name, -t weights type) from HF metadata.

Most converted builds need a "skeleton" source: an existing FLM NPU2 port of the
true base model that ships exactly the config/tokenizer/vision assets a build
requires (e.g. Atomic-Germ/Qwen3.5-4B-NPU2). Instead of asking the user for
-s/-o/-t, we follow the model card's ``base_model`` frontmatter chain upward and
stop at the first ancestor whose ``{org}/{basename}-NPU2`` mirror exists under a
preferred org. The chain also yields the display name and parameter-size token
for the output folder, and the pipeline tag tells us whether the finetune is a
VLM (convert language + vision weights).

Every network touch is optional: offline or private-repo failures degrade to a
``[WARN]`` and the caller keeps whatever the user passed explicitly.
"""
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .model_assets import _repo_id_from_url

# Orgs searched (in order) for the {basename}-NPU2 skeleton repo.
SKELETON_ORGS = ("Atomic-Germ", "FastFlowLM")
SKELETON_SUFFIX = "-NPU2"
MAX_CHAIN_DEPTH = 8

# Pipeline tags that imply vision-capable weights.
VLM_PIPELINE_TAGS = frozenset({
    "image-text-to-text",
    "image-to-text",
    "video-text-to-text",
    "any-to-any",
    "document-question-answering",
})
# Card tags with the same meaning (some cards skip the pipeline tag).
VLM_TAG_KEYWORDS = ("vision-language", "vlm", "image-text-to-text")

# Size token inside a skeleton name: 4B / 0.8B / 35B-A3B / E4B. Searched as a
# dash-delimited unit so "Claude" never matches, and kept in original case so
# lowercase sources stay lowercase (medgemma-1.5-4b-it -> 4b).
_SIZE_TOKEN_RE = re.compile(
    r"(?:^|-)(\d+(?:\.\d+)?[bB](?:-[Aa]\d+[bB])?|[Ee]\d+[bB])(?:$|-)"
)


@dataclass
class BuildPlan:
    """Everything derivable from an input repo's card metadata."""

    repo_id: str
    chain: List[str] = field(default_factory=list)
    skeleton: Optional[str] = None
    display_name: Optional[str] = None
    size_token: Optional[str] = None
    output_name: Optional[str] = None
    weights_type: str = "language"
    weights_reason: str = ""
    pipeline_tag: Optional[str] = None


def _model_info(repo_id: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot read HF metadata")
        return None
    try:
        return HfApi().model_info(repo_id)
    except Exception:
        return None


def _card_value(card, key):
    """Read a card field across huggingface_hub versions (dict-like or attrs)."""
    if card is None:
        return None
    try:
        return card[key]
    except Exception:
        return getattr(card, key, None)


def fetch_card(repo_id: str) -> dict:
    """The subset of a repo's card metadata used for derivation; {} if unavailable."""
    info = _model_info(repo_id)
    if info is None:
        return {}
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    return {
        "base_model": _card_value(card, "base_model"),
        "model_name": _card_value(card, "model_name"),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
    }


def normalize_base_models(value) -> List[str]:
    """base_model frontmatter to repo ids: string or list form, URLs flattened."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        item = _repo_id_from_url(item) or item
        if item and item not in result:
            result.append(item)
    return result


def walk_base_chain(
    repo_id: str,
    max_depth: int = MAX_CHAIN_DEPTH,
    cache: Optional[Dict[str, dict]] = None,
    fetch: Callable[[str], dict] = fetch_card,
) -> List[str]:
    """Ancestor chain [repo, parent, grandparent, ...] from base_model cards.

    Stops at depth, a cycle, or a repo without base_model. Note this does NOT
    stop at the "true" base: e.g. Qwen/Qwen3.5-4B itself declares
    Qwen/Qwen3.5-4B-Base. Callers pick the interesting node (find_skeleton).
    """
    if cache is None:
        cache = {}
    chain = [repo_id]
    seen = {repo_id}
    current = repo_id
    for _ in range(max_depth - 1):
        card = cache.get(current)
        if card is None:
            card = fetch(current)
            cache[current] = card
        parents = normalize_base_models(card.get("base_model"))
        if not parents:
            break
        nxt = parents[0]
        if nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return chain


def _skeleton_exists(candidate: str) -> bool:
    return _model_info(candidate) is not None


def find_skeleton(
    chain: List[str],
    orgs: List[str] = SKELETON_ORGS,
    probe: Callable[[str], bool] = _skeleton_exists,
) -> Optional[str]:
    """First ancestor with an {org}/{basename}-NPU2 mirror under a preferred org.

    Nearest ancestor wins, so a finetune of Qwen/Qwen3.5-4B resolves to
    Atomic-Germ/Qwen3.5-4B-NPU2 rather than walking all the way to -Base.
    """
    for node in chain:
        basename = os.path.basename(node.rstrip("/"))
        if basename.endswith(SKELETON_SUFFIX):
            continue
        for org in orgs:
            candidate = f"{org}/{basename}{SKELETON_SUFFIX}"
            if candidate == node:
                continue
            if probe(candidate):
                return candidate
    return None


def parse_size_token(name: str) -> Optional[str]:
    """Parameter-size token in a model/skeleton name (4B, 0.8B, 35B-A3B, E4B)."""
    stem = re.sub(rf"{SKELETON_SUFFIX}$", "", name, flags=re.IGNORECASE)
    match = _SIZE_TOKEN_RE.search(stem)
    return match.group(1) if match else None


def derive_display_name(card: dict, repo_id: str) -> str:
    """model_name frontmatter field, else the repo basename minus -GGUF."""
    name = str(card.get("model_name") or "").strip()
    if not name:
        name = os.path.basename(repo_id.rstrip("/"))
        name = re.sub(r"(?:-[Gg][Gg][Uu][Ff])+$", "", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name or repo_id


def infer_weights_type(card: dict) -> str:
    """'vision' when the card marks the model VLM, else 'language'."""
    pipeline = str(card.get("pipeline_tag") or "").lower()
    if pipeline in VLM_PIPELINE_TAGS:
        return "vision"
    tags = {str(t).lower() for t in card.get("tags") or []}
    if tags.intersection(VLM_TAG_KEYWORDS):
        return "vision"
    return "language"


def weights_type_reason(card: dict) -> str:
    """Human-readable signal behind infer_weights_type ('' when default)."""
    pipeline = str(card.get("pipeline_tag") or "").lower()
    if pipeline in VLM_PIPELINE_TAGS:
        return f"pipeline_tag {pipeline}"
    tags = {str(t).lower() for t in card.get("tags") or []}
    if tags.intersection(VLM_TAG_KEYWORDS):
        return "card tags"
    return ""


def format_chain(chain: List[str]) -> str:
    return " -> ".join(chain)


def display_name_for_arch(arch) -> Optional[str]:
    """Official-style base name for an arch whose name carries a size token.

    QWEN35_4B -> 'Qwen3.5-4B' (matches the {org}/{base}-NPU2 mirror naming).
    Returns None when the arch name has no size segment (plain 'qwen3',
    'gemma4', ...) because the resulting mirror name would be a guess.
    """
    from .constants import ModelArchNames

    names = ModelArchNames.get(arch) or []
    if not names:
        return None
    best = max(names, key=len)
    segments = re.split(r"[-_]", best)
    has_size = any(
        re.fullmatch(r"\d+(?:\.\d+)?b(?:-a\d+b)?|e\d+b", seg, re.IGNORECASE)
        for seg in segments
    )
    if not has_size:
        return None
    return "-".join(seg[:1].upper() + seg[1:] if seg[:1].isalpha() else seg for seg in segments)


def find_skeleton_for_arch(
    arch,
    orgs: List[str] = SKELETON_ORGS,
    probe: Callable[[str], bool] = _skeleton_exists,
) -> Optional[str]:
    """Skeleton mirror guessed from the architecture alone (no chain needed).

    Used when a repo declares no base_model anywhere: its config.json still
    reveals family+size, which maps to the same {org}/{base}-NPU2 convention.
    """
    display = display_name_for_arch(arch)
    if not display:
        return None
    for org in orgs:
        candidate = f"{org}/{display}{SKELETON_SUFFIX}"
        if probe(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Local README.md card parsing (-i <dir-with-README> / -i <file>.md)
# ---------------------------------------------------------------------------

_REPO_ID_RE = re.compile(
    r"(?:^|[\s\"'`(=])([A-Za-z][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)(?=[\s\"'`,.:;)\]]|$)"
)
# Code and CLI references rank a raw mention higher.
_REPO_ID_CONTEXT_RE = re.compile(
    r"(?:from_pretrained|AutoProcessor|AutoTokenizer|vllm serve|ollama run|"
    r"huggingface\.co/|hf\.co/|resolve/main)",
)
_NON_REPO_HINTS = ("cdn-uploads", "opensource.org", "img.shields.io", "github.com")


def parse_readme_frontmatter(text: str) -> dict:
    """Minimal YAML-frontmatter subset: flat keys plus simple lists.

    Tries PyYAML first (a transitive dep of huggingface_hub); falls back to a
    line-based parser that handles `key: value` and `- item` blocks, which is
    all model-card frontmatter needs.
    """
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped.startswith("---"):
        return {}
    end = stripped.find("\n---", 3)
    if end == -1:
        return {}
    block = stripped[3:end]
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(block)
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        pass
    data: dict = {}
    current_key = None
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith(("  ", "\t", "-")):
            item = line.strip().lstrip("-").strip()
            if current_key is not None and item:
                data.setdefault(current_key, [])
                if isinstance(data[current_key], list):
                    data[current_key].append(item)
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            data[key] = []
            current_key = key
        else:
            data[key] = value.strip("'\"")
            current_key = None
    return data


def extract_repo_ids(text: str) -> List[str]:
    """Rank org/name candidates mentioned in a model card body.

    Code snippets (from_pretrained/vllm/ollama) and links are strong signals;
    frequency breaks ties. URL-host strings (shields, cdn) are excluded.
    """
    scores: Counter = Counter()
    for match in _REPO_ID_RE.finditer(text):
        repo_id = match.group(1).strip(".,;")
        lowered = repo_id.lower()
        if any(hint in lowered for hint in _NON_REPO_HINTS):
            continue
        # org/name only; skip file paths like `very_long_document.txt/x`
        weight = 2 if _REPO_ID_CONTEXT_RE.search(text[max(0, match.start() - 40): match.end() + 40]) else 1
        scores[repo_id] += weight
    # Prefer the most-mentioned spelling but keep variants (case differences).
    ranked = [repo for repo, _ in scores.most_common()]
    return ranked


_PARAM_SIZE_RE = re.compile(
    r"(?:\bparams?(?:eters)?\s*[:=-]?\s*)?(\d+(?:\.\d+)?)\s*[bB]\b"
    r"|(\d+(?:\.\d+)?)\s*[- ]?billion\b",
)


def extract_param_size(text: str) -> Optional[str]:
    """Parameter-count claim from badge/text mentions ('4B', '4-billion')."""
    m = re.search(r"Parameters[-–]\s*(\d+(?:\.\d+)?)\s*B\b", text)
    if m:
        return m.group(1) + "B"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*[- ]?billion[- ]parameter", text, re.IGNORECASE)
    if m:
        return m.group(1) + "B"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*B\s+(?:param|model)", text, re.IGNORECASE)
    if m:
        return m.group(1) + "B"
    return None


def canonicalize_repo_id(repo_id: str) -> Optional[str]:
    """Resolve casing/prefix to the canonical HF id; None when unreachable."""
    info = _model_info(repo_id)
    if info is None:
        return None
    canonical = getattr(info, "id", None) or getattr(info, "modelId", None)
    return canonical or repo_id


def derive_build_plan_from_card(
    path: str,
    orgs: List[str] = SKELETON_ORGS,
    fetch: Callable[[str], dict] = fetch_card,
    probe: Callable[[str], bool] = _skeleton_exists,
    canonicalize: Callable[[str], Optional[str]] = canonicalize_repo_id,
) -> BuildPlan:
    """Derive a build plan from a local model card (directory or README file).

    The card itself supplies frontmatter (pipeline tag, tags, base_model);
    the body usually names the upstream repo id one or more times. That id is
    canonicalized against the Hub so the normal chain walk, skeleton lookup
    and name derivation apply as if the user had passed `-i <repo-id>`.
    """
    readme_path = path
    if os.path.isdir(path):
        readme_path = os.path.join(path, "README.md")
        if not os.path.isfile(readme_path):
            raise FileNotFoundError(f"No README.md in {path}")
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()

    frontmatter = parse_readme_frontmatter(text)
    display_hint = str(frontmatter.get("model_name") or "").strip()
    pipeline_tag = frontmatter.get("pipeline_tag")
    local_bases = normalize_base_models(frontmatter.get("base_model"))

    candidates = extract_repo_ids(text)
    plan = BuildPlan(repo_id="")
    primary = None
    for cand in candidates:
        canonical = canonicalize(cand)
        if canonical:
            primary = canonical
            break
    if primary is None and candidates:
        # Offline or unreachable: trust the most-mentioned spelling.
        primary = candidates[0]
        print(f"[WARN] Could not verify {primary} on the Hub; using it as-is")
    if primary is None and len(local_bases) == 1:
        # A frontmatter base_model with no body mention: treat it as the subject.
        primary = local_bases[0]

    card: Dict = {
        "base_model": frontmatter.get("base_model"),
        "model_name": frontmatter.get("model_name"),
        "pipeline_tag": pipeline_tag,
        "tags": frontmatter.get("tags") or [],
    }

    plan.repo_id = primary or ""
    plan.pipeline_tag = pipeline_tag
    plan.weights_type = infer_weights_type(card)
    plan.weights_reason = weights_type_reason(card)

    if primary:
        cache: Dict[str, dict] = {}
        remote_card = fetch(primary)
        cache[primary] = remote_card
        remote_chain = walk_base_chain(primary, cache=cache, fetch=fetch)
        if len(remote_chain) > 1:
            plan.chain = remote_chain
            plan.skeleton = find_skeleton(plan.chain, orgs, probe)
        else:
            # Remote card declares nothing either; seed the chain with any
            # locally-declared base_model before giving up.
            plan.chain = [primary] + local_bases
            plan.skeleton = find_skeleton(plan.chain, orgs, probe) if local_bases else None

    if not display_hint:
        if os.path.isdir(path):
            display_hint = os.path.basename(os.path.abspath(path))
        elif plan.repo_id:
            display_hint = os.path.basename(plan.repo_id)
        else:
            display_hint = "Model"
    plan.display_name = derive_display_name(
        {"model_name": display_hint}, plan.repo_id or display_hint
    )
    if not plan.size_token and plan.skeleton:
        plan.size_token = parse_size_token(os.path.basename(plan.skeleton))
    if not plan.size_token:
        plan.size_token = extract_param_size(text)
    parts = [plan.display_name or "Model"]
    if plan.size_token and plan.size_token.lower() not in {
        seg.lower() for seg in (plan.display_name or "").split("-")
    }:
        parts.append(plan.size_token)
    plan.output_name = "-".join(parts) + SKELETON_SUFFIX
    return plan


def derive_build_plan(
    repo_id: str,
    orgs: List[str] = SKELETON_ORGS,
    fetch: Callable[[str], dict] = fetch_card,
    probe: Callable[[str], bool] = _skeleton_exists,
) -> BuildPlan:
    """Resolve everything about a build that the input repo's metadata implies."""
    plan = BuildPlan(repo_id=repo_id)
    cache: Dict[str, dict] = {}
    plan.chain = walk_base_chain(repo_id, cache=cache, fetch=fetch)
    card = cache.get(repo_id) or {}
    plan.pipeline_tag = card.get("pipeline_tag")
    plan.weights_type = infer_weights_type(card)
    plan.weights_reason = weights_type_reason(card)
    plan.display_name = derive_display_name(card, repo_id)
    plan.skeleton = find_skeleton(plan.chain, orgs, probe)
    if plan.skeleton:
        plan.size_token = parse_size_token(os.path.basename(plan.skeleton))
    if plan.display_name:
        parts = [plan.display_name]
        # Don't duplicate a size the name already carries
        # (Qwen3.5-9B-Uncensored... already has 9B; medgemma-1.5-4b-it has 4b).
        segments = {seg.lower() for seg in plan.display_name.split("-")}
        if plan.size_token and plan.size_token.lower() not in segments:
            parts.append(plan.size_token)
        plan.output_name = "-".join(parts) + SKELETON_SUFFIX
    return plan
