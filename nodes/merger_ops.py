import torch
import scipy.ndimage
import torch.nn.functional as F

# ################# #
# MERGE OPERATORS
# ################# #
class Operation:
    def __init__(self, key, *sources):
        self.key = key
        self.sources = tuple(sources)
        self.merge_func = self.recurse
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
    def __init__(self, key, model_name, handlers, device, dtype):
        super().__init__(key)
        self.model_name = model_name
        self.handlers = handlers
        self.device = device
        self.dtype = dtype

    def merge(self) -> torch.Tensor:
        if self.model_name not in self.handlers:
            raise ValueError(f"Model '{self.model_name}' is required for this mode but was not provided.")
        return self.handlers[self.model_name].get_tensor(self.key).to(device=self.device, dtype=self.dtype)

class Multiply(Operation):
    def __init__(self, key, alpha, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha
    def oper(self, a) -> torch.Tensor:
        return a * self.alpha
class Add(Operation):
    def oper(self, a, b) -> torch.Tensor:
        return a + b
class Sub(Operation):
    def oper(self, a, b) -> torch.Tensor:
        return a - b

class SVD(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma = int(alpha), int(beta), gamma
    def oper(self, a, b) -> torch.Tensor:
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
        filtered = scipy.ndimage.median_filter(a.detach().cpu().to(torch.float32).numpy(), size=3)
        filtered = scipy.ndimage.gaussian_filter(filtered, sigma=1)
        return torch.tensor(filtered, dtype=a.dtype, device=a.device)
class TrainDiff(Operation):
    def oper(self, a, b, c) -> torch.Tensor:
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
        base_f = base.float() if base is not None else 0
        a_f, b_f = a.float() - base_f, b.float() - base_f
        c = torch.cosine_similarity(a_f.flatten(), b_f.flatten(), 0).clamp(-1, 1)
        d = ((c + 1) / 2)**self.gamma
        return (torch.lerp(a_f, b_f, self.alpha) * torch.lerp(d, 1 - d, self.beta)).to(a.dtype)
class Similarities(ExtractOp):
    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: return super().oper(None, a, b)
class PowerUpOp(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.seed = alpha, seed
    def oper(self, a, b):
        delta = b - a
        rng = torch.Generator(device=delta.device).manual_seed(self.seed)
        m = torch.bernoulli(torch.full_like(delta, 1 - self.alpha), generator=rng)
        return (m * delta) / (1 - self.alpha) if (1 - self.alpha) > 0 else torch.zeros_like(delta)
class InterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha, self.beta, self.gamma, self.seed = alpha, beta, gamma, seed
    def oper(self, a, b):
        alpha, delta = max(self.alpha, 0.001), torch.abs(a - b)
        max_delta = torch.max(delta)
        if max_delta == 0: return a
        diff = ((max_delta - delta) / max_delta if self.beta != 1 else delta / max_delta) ** (1 / alpha - 1)
        diff = torch.nan_to_num(diff)
        rng = torch.Generator(device=diff.device).manual_seed(self.seed)
        mask = torch.lerp(torch.bernoulli(torch.clamp(diff, 0, 1), generator=rng), diff, self.gamma)
        return a * (1 - mask) + b * mask

# ################# #
# CALCULATION MODES
# ################# #
class CalcMode:
    name = 'calcmode'; description = 'description'; models_used = []; param_docs = {}
    def create_recipe(self, key, **kwargs) -> Operation: raise NotImplementedError

class WeightSum(CalcMode):
    name = 'Weight-Sum'
    description = 'A * (1 - α) + B * α. Simple linear interpolation.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Interpolation weight. 0.0 is 100% Model A, 1.0 is 100% Model B.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        if kwargs['alpha'] >= 1: return b
        if kwargs['alpha'] <= 0: return a
        c = Multiply(key, 1 - kwargs['alpha'], a)
        d = Multiply(key, kwargs['alpha'], b)
        return Add(key, c, d)
class AddDifference(CalcMode):
    name = 'Add-Difference'
    description = 'A + (B - C) * α. Applies the difference between B and C to A.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Multiplier for the (B - C) difference.', 'beta': 'Smoothing (0=Off, 1=On).'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        c = LoadTensor(key, kwargs['model_c'], **loader_args)
        diff = Sub(key, b, c)
        if kwargs['beta'] == 1: diff = Smooth(key, diff)
        diffm = Multiply(key, kwargs['alpha'], diff)
        return Add(key, a, diffm)
class TrainDifference(CalcMode):
    name = 'Train-Difference'
    description = 'A + (B - C) * α, using tensor distances.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Multiplier for the calculated difference.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        c = LoadTensor(key, kwargs['model_c'], **loader_args)
        diff = TrainDiff(key, a, b, c)
        diffm = Multiply(key, kwargs['alpha'], diff)
        return Add(key, a, diffm)
class InterpDifference(CalcMode):
    name = 'Comparative-Interpolation'
    description = 'Interpolates A and B based on relative tensor value differences.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Curve strength.', 'beta': 'Style (0=Similarity, 1=Difference).', 'gamma': 'Mix (0=Random, 1=Linear).'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'): return LoadTensor(key, kwargs['model_a'], **loader_args)
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        return InterpolateDifference(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], kwargs['seed'], a, b)
class Extract(CalcMode):
    name = 'Extract-Features'
    description = 'Adds features from B and C to A, based on their similarity.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Weighting B vs C.', 'beta': 'Similarity vs dissimilarity.', 'gamma': 'Similarity bias.', 'delta': 'Multiplier for final features.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        c = LoadTensor(key, kwargs['model_c'], **loader_args)
        extracted = ExtractOp(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'] * 15, a, b, c)
        multiplied = Multiply(key, kwargs['delta'], extracted)
        return Add(key, a, multiplied)
class AddDisimilarity(CalcMode):
    name = 'Add-Dissimilarities'
    description = 'Adds dissimilar features between B and C to model A.'
    models_used = ['A', 'B', 'C']
    param_docs = {'alpha': 'Weighting B vs C.', 'beta': 'Multiplier for final features.', 'gamma': 'Similarity bias.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        c = LoadTensor(key, kwargs['model_c'], **loader_args)
        extracted = Similarities(key, kwargs['alpha'], 1, kwargs['gamma'] * 15, b, c)
        multiplied = Multiply(key, kwargs['beta'], extracted)
        return Add(key, a, multiplied)
class PowerUp(CalcMode):
    name = 'Power-Up (DARE)'
    description = 'Adds capabilities of B to A using Drop and Rescale (DARE).'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Dropout rate (proportion of weights from B to *drop*).', 'beta': 'Multiplier for the final difference.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        deltahat = PowerUpOp(key, kwargs['alpha'], kwargs['seed'], a, b)
        res = Multiply(key, kwargs['beta'], deltahat)
        return Add(key, a, res)
class SVD(CalcMode):
    name = 'SVD LoRA Extraction'
    description = 'Extracts the difference between B and A into a LoRA file.'
    models_used = ['A', 'B']
    param_docs = {'alpha': 'Rank for standard layers.', 'beta': 'Rank for 3x3 conv layers.', 'gamma': 'Clamp quantile for weights.'}
    def create_recipe(self, key, **kwargs):
        loader_args = {"handlers": kwargs['handlers'], "device": kwargs['device'], "dtype": kwargs['dtype']}
        a = LoadTensor(key, kwargs['model_a'], **loader_args)
        b = LoadTensor(key, kwargs['model_b'], **loader_args)
        return SVD(key, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], a, b)

TWO_MODEL_MODES = [WeightSum(), InterpDifference(), PowerUp(), SVD()]
THREE_MODEL_MODES = [AddDifference(), TrainDifference(), Extract(), AddDisimilarity()]