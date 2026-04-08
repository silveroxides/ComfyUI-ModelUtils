import os
import folder_paths
from comfy_api.latest import io
from .downloader_utils import scan_and_process

def get_subdirectories(folder_name: str) -> list[str]:
    """Helper to get a list of subdirectories for a given model folder name."""
    paths = folder_paths.get_folder_paths(folder_name)
    if not paths:
        return ["/"]

    subdirs = set()
    for base_path in paths:
        if not os.path.exists(base_path):
            continue
        for root, dirs, _ in os.walk(base_path):
            # Ignore hidden directories directly during walk
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            rel_path = os.path.relpath(root, base_path)
            if rel_path == ".":
                subdirs.add("/")
            else:
                # Store with forward slashes for consistency
                subdirs.add("/" + rel_path.replace("\\", "/"))

    return sorted(list(subdirs))

class BaseInfoMetaDownloaderNode(io.ComfyNode):
    CATEGORY = "ModelUtils/Downloader"
    OUTPUT_NODE = True

    @classmethod
    def get_common_inputs(cls):
        return {
            "recursive": io.Boolean.Input("recursive", default=True, tooltip="Whether to scan subdirectories recursively."),
            "nsfw_level": io.Combo.Input("nsfw_level", options=["None", "Soft", "Mature", "X", "XXX", "All"], default="All", tooltip="Filter models based on NSFW level in metadata."),
            "max_examples": io.Int.Input("max_examples", default=0, min=0, max=100, tooltip="Maximum number of examples to process."),
            "threads": io.Int.Input("threads", default=4, min=1, max=32, tooltip="Number of threads to use for processing."),
            "api_key": io.String.Input("api_key", default="", tooltip="API key for accessing the model repository.")
        }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name=cls.__name__.replace("Downloader", " Downloader"),
            category=cls.CATEGORY,
            inputs=list(cls.get_common_inputs().values()),
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
            description=("Scans specified directories for models, extracts metadata, generates previews, and optionally compresses files.")
        )

    @classmethod
    def execute(cls, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        scan_dirs = cls.get_scan_dirs(**kwargs)
        if not scan_dirs:
            return io.NodeOutput("No directories to scan.")

        # Call the scan_and_process function
        stats = scan_and_process(
            scan_dirs=scan_dirs,
            recursive=recursive,
            nsfw_level=nsfw_level,
            max_examples=max_examples,
            api_key=api_key if api_key else None,
            threads=threads
        )

        mb_saved = stats.get("space_saved", 0) / (1024 * 1024)

        output_msg = (
            f"Scan Complete for {len(scan_dirs)} directory/directories.\n"
            f"Total Models Processed: {stats.get('total_processed', 0)}\n"
            f"New Models Hashed: {stats.get('new_hashes', 0)}\n"
            f"Images Processed: {stats.get('images_processed', 0)}\n"
            f"Workflows Extracted: {stats.get('workflows_extracted', 0)}\n"
            f"Space Saved via Compression: {mb_saved:.2f} MB"
        )

        return io.NodeOutput(output_msg)

    @classmethod
    def get_scan_dirs(cls, **kwargs) -> list[str]:
        return []

class CheckpointInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("checkpoints"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="Checkpoint Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("checkpoints")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class DiffusionModelInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("diffusion_models"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="Diffusion Model Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("diffusion_models")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class LoRAInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("loras"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="LoRA Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("loras")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class EmbeddingInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("embeddings"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="Embedding Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("embeddings")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class VAEInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("vae"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="VAE Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("vae")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class ControlNetInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.insert(0, io.Combo.Input("subdirectory", options=get_subdirectories("controlnet"), default="/"))
        return io.Schema(
            node_id=cls.__name__,
            display_name="ControlNet Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, subdirectory, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        return super().execute(
            recursive=recursive, nsfw_level=nsfw_level, max_examples=max_examples,
            threads=threads, api_key=api_key, subdirectory=subdirectory, **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        base_paths = folder_paths.get_folder_paths("controlnet")
        if not base_paths:
            return []
        sub = kwargs.get("subdirectory", "/")
        if sub == "/":
            return base_paths

        target_paths = []
        for bp in base_paths:
            target_path = os.path.join(bp, sub.lstrip("/"))
            if os.path.exists(target_path):
                target_paths.append(target_path)
        return target_paths

class ManualPathInfoMetaDownloader(BaseInfoMetaDownloaderNode):
    @classmethod
    def define_schema(cls):
        inputs = list(cls.get_common_inputs().values())
        inputs.append(io.String.Input("scan_dir", default=""))
        return io.Schema(
            node_id=cls.__name__,
            display_name="Manual Path Downloader",
            category=cls.CATEGORY,
            inputs=inputs,
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, recursive, nsfw_level, max_examples, threads, api_key, scan_dir, **kwargs):
        return super().execute(
            recursive=recursive,
            nsfw_level=nsfw_level,
            max_examples=max_examples,
            threads=threads,
            api_key=api_key,
            scan_dir=scan_dir,
            **kwargs
        )

    @classmethod
    def get_scan_dirs(cls, **kwargs):
        scan_dir = kwargs.get("scan_dir", "")
        if scan_dir.strip():
            return [scan_dir.strip()]
        return []
