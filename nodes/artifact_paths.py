import os

import folder_paths


CANONICAL_MODEL_CATEGORIES = frozenset(
    {"checkpoints", "diffusion_models", "text_encoders", "loras", "embeddings"}
)


def canonical_model_artifact_path(category: str, output_filename: str) -> tuple[str, str]:
    """Return the canonical write path and category-relative loader value."""
    if category not in CANONICAL_MODEL_CATEGORIES:
        raise ValueError(f"Unsupported model artifact category: {category}")

    relative_name = output_filename.strip().replace("\\", "/")
    if not relative_name:
        raise ValueError("Output filename must not be empty.")
    if relative_name.startswith("/") or os.path.isabs(relative_name):
        raise ValueError("Output filename must be relative to its model category.")
    while relative_name.lower().endswith(".safetensors"):
        relative_name = relative_name[:-12]
    if not relative_name:
        raise ValueError("Output filename must include a name before its extension.")
    relative_name = f"{relative_name}.safetensors"

    category_root = os.path.abspath(os.path.join(folder_paths.models_dir, category))
    output_path = os.path.abspath(
        os.path.join(category_root, *relative_name.split("/"))
    )
    if os.path.commonpath((category_root, output_path)) != category_root:
        raise ValueError("Output filename must stay inside its model category directory.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_name = os.path.relpath(output_path, category_root).replace(os.sep, "/")
    return output_path, output_name
