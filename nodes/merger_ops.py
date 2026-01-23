import torch
import scipy.ndimage
import torch.nn.functional as F
from enum import Enum


class MissingTensorBehavior(Enum):
    """Controls behavior when a tensor key is missing from a model."""
    ERROR = "error"      # Raise exception (strict mode)
    SKIP = "skip"        # Return None, let caller handle
    ZEROS = "zeros"      # Return zeros tensor matching shape from primary model


class MissingTensorError(Exception):
    """Raised when a tensor is missing and behavior is ERROR."""
    pass

# ################# #
# UTILITIES
# ################# #
def resize_by_interpolation(tensor1, tensor2):
    """Resizes tensor2 to match tensor1's shape using interpolation."""
    if tensor1 is None or tensor2 is None:
        return tensor1, tensor2
    if tensor1.shape == tensor2.shape:
        return tensor1, tensor2

    dims1 = len(tensor1.shape)
    dims2 = len(tensor2.shape)

    if dims1 != dims2 or dims1 > 4 or dims1 == 0:
        return tensor1, tensor2

    orig_dtype = tensor2.dtype
    t2_f = tensor2.float()

    # F.interpolate expects [B, C, ...]
    # We treat the tensor as [1, 1, ...dims]
    view_shape = [1, 1] + list(tensor2.shape)
    t2_f = t2_f.view(view_shape)

    # Select mode based on dimensions
    if dims1 == 1:
        mode = 'linear'
    elif dims1 == 2:
        mode = 'bilinear'
    else:
        mode = 'trilinear'

    t2_res = F.interpolate(t2_f, size=list(tensor1.shape), mode=mode, align_corners=False)

    return tensor1, t2_res.view(tensor1.shape).to(orig_dtype)


def resize_tensors(tensor1, tensor2, mode='pad/crop'):
    """Resizes tensors to match shapes using either padding or interpolation."""
    if tensor1 is None or tensor2 is None:
        return tensor1, tensor2
    if tensor1.shape == tensor2.shape:
        return tensor1, tensor2

    if mode == 'interpolate':
        return resize_by_interpolation(tensor1, tensor2)

    dims1 = len(tensor1.shape)
    dims2 = len(tensor2.shape)

    if dims1 != dims2 or dims1 > 4 or dims1 == 0:
        return tensor1, tensor2

    padding1 = []
    padding2 = []

    # F.pad expects: (last_dim_left, last_dim_right, prev_dim_top, prev_dim_bottom, ...)
    for i in range(dims1 - 1, -1, -1):
        s1 = tensor1.shape[i]
        s2 = tensor2.shape[i]
        if s1 < s2:
            padding1.extend([0, s2 - s1])
            padding2.extend([0, 0])
        elif s2 < s1:
            padding1.extend([0, 0])
            padding2.extend([0, s1 - s2])
        else:
            padding1.extend([0, 0])
            padding2.extend([0, 0])

    if any(p > 0 for p in padding1):
        tensor1 = F.pad(tensor1, tuple(padding1))
    if any(p > 0 for p in padding2):
        tensor2 = F.pad(tensor2, tuple(padding2))

    return tensor1, tensor2


# ################# #
# MERGE OPERATORS
# ################# #
class Operation:
    def __init__(self, key, *sources):
        self.key = key
        self.sources = tuple(sources)
        self.merge_func = self.recurse
        self.alignment_mode = 'pad/crop'
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.seed = None

    def __eq__(self, other):
        return (type(self), self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources) == \
               (type(other), other.key, other.alpha, other.beta, other.gamma, other.delta, other.seed, other.sources)

    def __hash__(self):
        return hash((type(self), self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources))

    def oper(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def recurse(self, operation):
        source_tensors = [source_oper.merge() for source_oper in operation.sources]
        return operation.oper(*source_tensors)

    def merge(self):
        return self.merge_func(self)

class LoadTensor(Operation):
    def __init__(self, key, model_name, handlers, device, dtype,
                 on_missing=MissingTensorBehavior.ERROR, fallback_shape=None, fallback_dtype=None):
        super().__init__(key)
        self.model_name = model_name
        self.handlers = handlers
        self.device = device
        self.dtype = dtype
        self.on_missing = on_missing
        self.fallback_shape = fallback_shape
        self.fallback_dtype = fallback_dtype

    def merge(self) -> torch.Tensor:
        if self.model_name not in self.handlers:
            raise ValueError(f"Model '{self.model_name}' is required for this mode but was not provided.")

        handler = self.handlers[self.model_name]
        if self.key not in handler.keys():
            if self.on_missing == MissingTensorBehavior.ERROR:
                raise MissingTensorError(
                    f"Key '{self.key}' not found in model '{self.model_name}'"
                )
            elif self.on_missing == MissingTensorBehavior.SKIP:
                return None
            elif self.on_missing == MissingTensorBehavior.ZEROS:
                if self.fallback_shape is None:
                    raise ValueError(
                        f"Cannot create zeros for '{self.key}' without fallback_shape"
                    )
                dtype = self.fallback_dtype if self.fallback_dtype else self.dtype
                return torch.zeros(self.fallback_shape, device=self.device, dtype=dtype)

        return handler.get_tensor(self.key).to(device=self.device, dtype=self.dtype)

class Multiply(Operation):
    def __init__(self, key, alpha, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha
    def oper(self, a) -> torch.Tensor:
        if a is None:
            return None
        return a * self.alpha


class Add(Operation):
    def oper(self, a, b) -> torch.Tensor:
        if a is None:
            return b
        if b is None:
            return a
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        return a + b


class Sub(Operation):
    def oper(self, a, b) -> torch.Tensor:
        if a is None or b is None:
            return None  # Cannot compute difference with missing operand
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        return a - b

class SVD(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources, target_shape=None):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma = int(alpha), int(beta), gamma
        self.target_shape = target_shape

    def oper(self, a, b) -> torch.Tensor:
        if a is None or b is None:
            return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)

        # For SVD extraction, we must ensure the diff matches the original Model A shape
        # so the resulting LoRA weights are compatible with the target architecture.
        if self.target_shape and a.shape != self.target_shape:
            if self.alignment_mode == 'interpolate':
                # Re-align both to match target_shape via interpolation
                dummy = torch.empty(self.target_shape, device=a.device)
                _, a = resize_by_interpolation(dummy, a)
                _, b = resize_by_interpolation(dummy, b)
            else:
                slices = tuple(slice(0, min(res_s, tgt_s)) for res_s, tgt_s in zip(a.shape, self.target_shape))
                a = a[slices]
                b = b[slices]

        diff, weights, conv2d = a - b, {}, len(a.size()) == 4
        kernel_size = None if not conv2d else a.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        rank = self.alpha if not conv2d_3x3 or self.beta is None else self.beta
        rank = min(rank, a.size()[1], a.size()[0])
        matrix = diff.flatten(start_dim=1) if conv2d_3x3 else diff.squeeze() if conv2d else diff
        U, S, Vh = torch.linalg.svd(matrix)
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
        U = U @ torch.diag(S)
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, self.gamma)
        U, Vh = U.clamp(-hi_val, hi_val), Vh.clamp(-hi_val, hi_val)
        if conv2d:
            Vh = Vh.reshape(rank, a.size()[1], *kernel_size) if conv2d_3x3 else Vh.reshape(rank, a.size()[1], 1, 1)
            U = U.reshape(a.size()[0], rank, 1, 1)
        weights_sd = {f"{self.key}.lora_up.weight": U, f"{self.key}.lora_down.weight": Vh, f"{self.key}.alpha": torch.tensor(float(rank))}
        return weights_sd

class Smooth(Operation):
    def oper(self, a) -> torch.Tensor:
        if a is None:
            return None
        filtered = scipy.ndimage.median_filter(a.detach().cpu().to(torch.float32).numpy(), size=3)
        filtered = scipy.ndimage.gaussian_filter(filtered, sigma=1)
        return torch.tensor(filtered, dtype=a.dtype, device=a.device)
class TrainDiff(Operation):
    def oper(self, a, b, c) -> torch.Tensor:
        if a is None or b is None or c is None:
            return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        a, c = resize_tensors(a, c, mode=self.alignment_mode)
        b, c = resize_tensors(b, c, mode=self.alignment_mode)

        if torch.allclose(b.float(), c.float()): return torch.zeros_like(a)
        diff_bc = b.float() - c.float()
        dist_bc, dist_ab = torch.abs(diff_bc), torch.abs(b.float() - a.float())
        scale = torch.where(dist_bc + dist_ab != 0, dist_ab / (dist_bc + dist_ab), 0.0)
        return (torch.sign(diff_bc) * torch.abs(scale) * torch.abs(diff_bc)).to(a.dtype) * 1.8
class ExtractOp(Operation):
    def __init__(self, key, alpha, beta, gamma, *args):
        super().__init__(key, *args)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    def oper(self, base: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a is None or b is None:
            return None
        if base is not None:
            base, a = resize_tensors(base, a, mode=self.alignment_mode)
            base, b = resize_tensors(base, b, mode=self.alignment_mode)
        a, b = resize_tensors(a, b, mode=self.alignment_mode)

        dtype = base.dtype if base is not None else a.dtype
        base_f = base.float() if base is not None else 0
        a_f, b_f = a.float() - base_f, b.float() - base_f
        c = torch.cosine_similarity(a_f, b_f, -1).clamp(-1, 1).unsqueeze(-1)
        d = ((c + 1) / 2)**self.gamma
        return (torch.lerp(a_f, b_f, self.alpha) * torch.lerp(d, 1 - d, self.beta)).to(dtype)


class Similarities(ExtractOp):
    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a is None or b is None:
            return None
        return super().oper(None, a, b)
class PowerUpOp(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.seed = alpha, seed
    def oper(self, a, b):
        if a is None or b is None:
            return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        delta = b - a
        keep_prob = 1 - self.alpha
        if keep_prob <= 0:
            return torch.zeros_like(delta)
        rng = torch.Generator(device=delta.device).manual_seed(self.seed)
        m = (torch.empty_like(delta).uniform_(0, 1, generator=rng) < keep_prob).to(delta.dtype)
        return (m * delta) / keep_prob


class InterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma, self.seed = alpha, beta, gamma, seed
    def oper(self, a, b):
        if a is None or b is None:
            return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        alpha, delta = max(self.alpha, 0.001), torch.abs(a - b)
        max_delta = torch.max(delta)
        if max_delta == 0: return a
        diff = ((max_delta - delta) / max_delta if self.beta != 1 else delta / max_delta) ** (1 / alpha - 1)
        diff = torch.nan_to_num(diff)
        rng = torch.Generator(device=diff.device).manual_seed(self.seed)
        mask = torch.lerp(torch.bernoulli(torch.clamp(diff, 0, 1), generator=rng), diff, self.gamma)
        return a * (1 - mask) + b * mask

class ManualEnhancedInterpolateDifferenceOp(Operation):
    def __init__(self, key, alpha, beta, gamma, delta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma, self.delta, self.seed = alpha, beta, gamma, delta, seed

    def oper(self, a, b):
        if a is None or b is None: return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        delta = torch.abs(a - b)
        diff = torch.nan_to_num((torch.max(delta) - delta) / torch.max(delta))
        mean_diff = torch.mean(diff, 0, keepdim=True)
        mask = torch.logical_and(self.beta < mean_diff, mean_diff < self.gamma)
        powered_diff = torch.nan_to_num(diff ** (1 / max(self.alpha, 0.001) - 1))
        masked_diff = powered_diff * mask.float()
        rng = torch.Generator(device=a.device).manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.delta)
        return a * (1 - interpolated_mask) + b * interpolated_mask


class AutoEnhancedInterpolateDifferenceOp(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma, self.seed = alpha, beta, gamma, seed

    def oper(self, a, b):
        if a is None or b is None: return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        delta = torch.abs(a - b)
        max_delta = torch.max(delta)
        diff = torch.nan_to_num((max_delta - delta) / max_delta)
        mean_diff = torch.mean(diff)
        lower_threshold = mean_diff * (1 - self.beta)
        upper_threshold = mean_diff * (1 + self.beta)
        mask = torch.logical_and(lower_threshold < diff, diff < upper_threshold)
        powered_diff = torch.nan_to_num(diff ** (1 / max(self.alpha, 0.001) - 1))
        masked_diff = powered_diff * mask.float()
        rng = torch.Generator(device=a.device).manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.gamma)
        return a * (1 - interpolated_mask) + b * interpolated_mask


class WeightSumCutoffOp(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def oper(self, a, b):
        if a is None or b is None: return None
        a, b = resize_tensors(a, b, mode=self.alignment_mode)
        delta = torch.abs(a - b)
        diff = torch.nan_to_num((torch.max(delta) - delta) / torch.max(delta))
        mean = torch.mean(diff, 0, True)
        mask = torch.logical_and(mean < self.beta, self.gamma < mean)
        mul = self.alpha * mask
        return a * (1 - mul) + b * mul


# ################# #
# CALCULATION MODES
# ################# #
class CalcMode:
    name = 'calcmode'; description = 'description'; models_used = []; param_docs = {}
    def create_recipe(self, key, **kwargs) -> Operation: raise NotImplementedError

    @staticmethod
    def _get_secondary_loader_args(kwargs, mismatch_mode):
        """Helper to build loader args for secondary models (B, C) with fallback support."""
        base_args = {
            "handlers": kwargs['handlers'],
            "device": kwargs['device'],
            "dtype": kwargs['dtype'],
            "on_missing": mismatch_mode,
        }
        # Add fallback info for zeros mode
        if '_tensor_a_shape' in kwargs:
            base_args['fallback_shape'] = kwargs['_tensor_a_shape']
            base_args['fallback_dtype'] = kwargs.get('_tensor_a_dtype', kwargs['dtype'])
        return base_args


class PreloadedTensor(Operation):
    """Returns a pre-loaded tensor instead of loading from file.

    Used when Model A's tensor is pre-loaded in the merge loop.
    """
    def __init__(self, key, tensor):
        super().__init__(key)
        self.tensor = tensor

    def merge(self) -> torch.Tensor:
        return self.tensor

class WeightSum(CalcMode):
    name = 'Weight-Sum'
    description = 'A * (1 - α) + B * α. Simple linear interpolation.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Interpolation weight. 0.0 is 100% Model A, 1.0 is 100% Model B.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        # Use pre-loaded tensor_a if available
        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)

        if kwargs['alpha'] >= 1: return b
        if kwargs['alpha'] <= 0: return a
        c = Multiply(key, 1 - kwargs['alpha'], a)
        d = Multiply(key, kwargs['alpha'], b)
        res = Add(key, c, d)
        res.alignment_mode = alignment_mode
        return res


class AddDifference(CalcMode):
    name = 'Add-Difference'
    description = 'A + (B - C) * α. Applies the difference between B and C to A.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Multiplier for the (B - C) difference.', 'beta': 'Smoothing (0=Off, 1=On).'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_bc = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_bc)
        c = LoadTensor(key, kwargs['model_c'], **loader_args_bc)
        diff = Sub(key, b, c)
        if kwargs['beta'] == 1: diff = Smooth(key, diff)
        diffm = Multiply(key, kwargs['alpha'], diff)
        res = Add(key, a, diffm)
        res.alignment_mode = alignment_mode
        return res


class TrainDifference(CalcMode):
    name = 'Train-Difference'
    description = 'A + (B - C) * α, using tensor distances.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Multiplier for the calculated difference.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_bc = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_bc)
        c = LoadTensor(key, kwargs['model_c'], **loader_args_bc)
        diff = TrainDiff(key, a, b, c)
        diffm = Multiply(key, kwargs['alpha'], diff)
        res = Add(key, a, diffm)
        res.alignment_mode = alignment_mode
        return res
class InterpDifference(CalcMode):
    name = 'Comparative-Interpolation'
    description = 'Interpolates A and B based on relative tensor value differences.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Curve strength.', 'beta': 'Style (0=Similarity, 1=Difference).', 'gamma': 'Mix (0=Random, 1=Linear).'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        # Skip certain keys entirely
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        res = InterpolateDifference(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['seed'], a, b)
        res.alignment_mode = alignment_mode
        return res


class Extract(CalcMode):
    name = 'Extract-Features'
    description = 'Adds features from B and C to A, based on their similarity.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Weighting B vs C.', 'beta': 'Similarity vs dissimilarity.', 'gamma': 'Similarity bias.', 'delta': 'Multiplier for final features.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_bc = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_bc)
        c = LoadTensor(key, kwargs['model_c'], **loader_args_bc)
        extracted = ExtractOp(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'] * 15, a, b, c)
        multiplied = Multiply(key, kwargs['delta'], extracted)
        res = Add(key, a, multiplied)
        res.alignment_mode = alignment_mode
        return res


class AddDisimilarity(CalcMode):
    name = 'Add-Dissimilarities'
    description = 'Adds dissimilar features between B and C to model A.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Weighting B vs C.', 'beta': 'Multiplier for final features.', 'gamma': 'Similarity bias.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_bc = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_bc)
        c = LoadTensor(key, kwargs['model_c'], **loader_args_bc)
        extracted = Similarities(key, kwargs['alpha'], 1, kwargs['gamma'] * 15, b, c)
        multiplied = Multiply(key, kwargs['beta'], extracted)
        res = Add(key, a, multiplied)
        res.alignment_mode = alignment_mode
        return res


class PowerUp(CalcMode):
    name = 'Power-Up (DARE)'
    description = 'Adds capabilities of B to A using Drop and Rescale (DARE).'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Dropout rate (proportion of weights from B to *drop*).', 'beta': 'Multiplier for the final difference.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        deltahat = PowerUpOp(key, kwargs['alpha'], kwargs['seed'], a, b)
        deltahat.alignment_mode = alignment_mode # Important!
        multiplied = Multiply(key, kwargs['beta'], deltahat)
        res = Add(key, a, multiplied)
        res.alignment_mode = alignment_mode
        return res


class SVDMode(CalcMode):
    name = 'SVD LoRA Extraction'
    description = 'Extracts the difference between B and A into a LoRA file.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Rank for standard layers.', 'beta': 'Rank for 3x3 conv layers.', 'gamma': 'Clamp quantile for weights.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        res = SVD(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], a, b, target_shape=kwargs.get('_tensor_a_shape'))
        res.alignment_mode = alignment_mode
        return res


class ManEnhInterpDifference(CalcMode):
    name = 'Enhanced Man Interp'
    description = 'Enhanced interpolation between each pair of values from A and B depending on their difference relative to other values.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Interpolation strength.', 'beta': 'Lower mean threshold.', 'gamma': 'Upper mean threshold.', 'delta': 'Smoothness factor.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        res = ManualEnhancedInterpolateDifferenceOp(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['delta'], kwargs['seed'], a, b)
        res.alignment_mode = alignment_mode
        return res


class AutoEnhInterpDifference(CalcMode):
    name = 'Enhanced Auto Interp'
    description = 'Interpolates between each pair of values from A and B depending on their difference relative to other values.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Interpolation strength.', 'beta': 'Threshold adjustment factor.', 'gamma': 'Smoothness factor.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        res = AutoEnhancedInterpolateDifferenceOp(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['seed'], a, b)
        res.alignment_mode = alignment_mode
        return res


class WeightSumCutoffMode(CalcMode):
    name = 'Weight-Sum Cutoff'
    description = 'Weight-sum with cutoff based on value differences.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Interpolation weight.', 'beta': 'Upper threshold.', 'gamma': 'Lower threshold.'}
    def create_recipe(self, key, **kwargs):
        mismatch_mode = kwargs.get('mismatch_mode', MissingTensorBehavior.SKIP)
        alignment_mode = kwargs.get('alignment_mode', 'pad/crop')

        if '_tensor_a' in kwargs:
            a = PreloadedTensor(key, kwargs['_tensor_a'])
        else:
            loader_args_a = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
            a = LoadTensor(key, kwargs['model_a'], **loader_args_a)

        loader_args_b = self._get_secondary_loader_args(kwargs, mismatch_mode)
        b = LoadTensor(key, kwargs['model_b'], **loader_args_b)
        res = WeightSumCutoffOp(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], a, b)
        res.alignment_mode = alignment_mode
        return res


TWO_MODEL_MODES = [WeightSum(), InterpDifference(), PowerUp(), SVDMode(), ManEnhInterpDifference(), AutoEnhInterpDifference(), WeightSumCutoffMode()]
THREE_MODEL_MODES = [AddDifference(), TrainDifference(), Extract(), AddDisimilarity()]