import folder_paths
from comfy_api.latest import io
from .downloader_utils import scan_and_process

class BaseDownloaderNode(io.ComfyNode):
    CATEGORY = "ModelUtils/Downloader"
    OUTPUT_NODE = True

    @classmethod
    def get_common_inputs(cls):
        return {
            "recursive": io.BOOLEAN(default=True),
            "nsfw_level": io.COMBO(["None", "Soft", "Mature", "X", "XXX", "All"], default="All"),
            "max_examples": io.INT(default=0, min=0, max=100),
            "threads": io.INT(default=4, min=1, max=32),
            "api_key": io.STRING(default="")
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": cls.get_common_inputs()}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process"

    def process(self, recursive, nsfw_level, max_examples, threads, api_key, **kwargs):
        scan_dirs = self.get_scan_dirs(**kwargs)
        if not scan_dirs:
            return ("No directories to scan.",)
        
        # Call the scan_and_process function
        scan_and_process(
            scan_dirs=scan_dirs,
            recursive=recursive,
            nsfw_level=nsfw_level,
            max_examples=max_examples,
            api_key=api_key if api_key else None,
            threads=threads
        )
        
        return (f"Completed scanning and downloading for {len(scan_dirs)} directories.",)

    def get_scan_dirs(self, **kwargs) -> list[str]:
        return []


class CheckpointDownloader(BaseDownloaderNode):
    def get_scan_dirs(self, **kwargs):
        return folder_paths.get_folder_paths("checkpoints")


class LoRADownloader(BaseDownloaderNode):
    def get_scan_dirs(self, **kwargs):
        return folder_paths.get_folder_paths("loras")


class EmbeddingDownloader(BaseDownloaderNode):
    def get_scan_dirs(self, **kwargs):
        return folder_paths.get_folder_paths("embeddings")


class VAEDownloader(BaseDownloaderNode):
    def get_scan_dirs(self, **kwargs):
        return folder_paths.get_folder_paths("vae")


class ControlNetDownloader(BaseDownloaderNode):
    def get_scan_dirs(self, **kwargs):
        return folder_paths.get_folder_paths("controlnet")


class ManualPathDownloader(BaseDownloaderNode):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls.get_common_inputs()
        inputs["scan_dir"] = io.STRING(default="")
        return {"required": inputs}

    def get_scan_dirs(self, scan_dir, **kwargs):
        if scan_dir.strip():
            return [scan_dir.strip()]
        return []
