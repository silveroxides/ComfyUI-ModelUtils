import glob
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import folder_paths
from comfy_api.latest import io


def get_potential_preview_files(base_path: str) -> list[str]:
    exts = ["png", "jpg", "jpeg", "webp", "mp4"]
    files = []
    for ext in exts:
        files.append(f"{base_path}.preview.{ext}")
        files.append(f"{base_path}.{ext}")
    return files


def load_image_tensor(path: str):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def get_model_workflows(base_path: str) -> list[str]:
    workflows = []
    main_json = f"{base_path}.json"
    if os.path.exists(main_json):
        workflows.append(main_json)

    preview_json = f"{base_path}.preview.json"
    if os.path.exists(preview_json) and preview_json not in workflows:
        workflows.append(preview_json)

    example_jsons = glob.glob(f"{base_path}.example.*.json")
    example_jsons.sort()
    for ej in example_jsons:
        if ej not in workflows:
            workflows.append(ej)

    return workflows


def get_model_metadata_file(base_path: str) -> str | None:
    civitai_info = f"{base_path}.civitai.info"
    if os.path.exists(civitai_info):
        return civitai_info

    metadata_json = f"{base_path}.metadata.json"
    if os.path.exists(metadata_json):
        return metadata_json

    return None

class BaseModelInfoLoader(io.ComfyNode):
    CATEGORY = "ModelUtils/Info"

    @classmethod
    def get_info(cls, category, name, workflow_index):
        full_path = folder_paths.get_full_path(category, name)
        if not full_path or not os.path.exists(full_path):
            return (None, "{}", "{}")

        base_path = os.path.splitext(full_path)[0]

        # 1. Image Preview
        image_tensor = None
        previews = get_potential_preview_files(base_path)
        for p in previews:
            if os.path.exists(p) and not p.lower().endswith('.mp4'):
                image_tensor = load_image_tensor(p)
                if image_tensor is not None:
                    break

        # 2. Workflow
        workflow_json = "{}"
        workflows = get_model_workflows(base_path)
        if workflows and 0 <= workflow_index < len(workflows):
            try:
                with open(workflows[workflow_index], 'r', encoding='utf-8') as f:
                    workflow_json = f.read()
            except Exception as e:
                print(f"Error reading workflow {workflows[workflow_index]}: {e}")

        # 3. Metadata
        metadata_json = "{}"
        metadata_file = get_model_metadata_file(base_path)
        if metadata_file:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_json = f.read()
            except Exception as e:
                print(f"Error reading metadata {metadata_file}: {e}")

        return (image_tensor, workflow_json, metadata_json)

class CheckpointInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="Checkpoint Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("checkpoints"), tooltip="Checkpoint whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, checkpoint, workflow_index):
        res = cls.get_info("checkpoints", checkpoint, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])

class LoRAInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="LoRA Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("lora", options=folder_paths.get_filename_list("loras"), tooltip="LoRA whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, lora, workflow_index):
        res = cls.get_info("loras", lora, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])

class EmbeddingInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="Embedding Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("embedding", options=folder_paths.get_filename_list("embeddings"), tooltip="Embedding whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, embedding, workflow_index):
        res = cls.get_info("embeddings", embedding, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])

class VAEInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="VAE Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("vae", options=folder_paths.get_filename_list("vae"), tooltip="VAE whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, vae, workflow_index):
        res = cls.get_info("vae", vae, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])

class ControlNetInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="ControlNet Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("controlnet", options=folder_paths.get_filename_list("controlnet"), tooltip="ControlNet whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, controlnet, workflow_index):
        res = cls.get_info("controlnet", controlnet, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])

class DiffusionModelInfoLoader(BaseModelInfoLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="Diffusion Model Info Loader",
            category=cls.CATEGORY,
            inputs=[
                io.Combo.Input("diffusion_model", options=folder_paths.get_filename_list("diffusion_models"), tooltip="Diffusion model whose downloaded information and example workflow will be returned."),
                io.Int.Input("workflow_index", default=0, min=0, max=100, tooltip="Zero-based index of the saved example workflow to return when multiple workflows are available.")
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="metadata_json")
            ]
        )

    @classmethod
    def execute(cls, diffusion_model, workflow_index):
        res = cls.get_info("diffusion_models", diffusion_model, workflow_index)
        return io.NodeOutput(res[0], res[1], res[2])
