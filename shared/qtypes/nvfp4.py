# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import ast
import os
import torch
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes

try:
    from torch._subclasses.fake_tensor import FakeTensor as _TorchFakeTensor
except Exception:  # pragma: no cover
    _TorchFakeTensor = ()

try:
    from torch._subclasses.functional_tensor import FunctionalTensor as _TorchFunctionalTensor
except Exception:  # pragma: no cover
    _TorchFunctionalTensor = ()

def _maybe_add_nvfp4_cu13_dll_dir():
    if os.name != "nt":
        return
    try:
        import nvidia.cu13
        dll_dir = os.path.join(nvidia.cu13.__path__[0], "bin", "x86_64")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
    except Exception:
        pass

try:
    from comfy_kitchen.backends import cuda as _ck_cuda
    _ck_cuda_available = getattr(_ck_cuda, "_EXT_AVAILABLE", False) and torch.version.hip is None  # under ROCm it loads but takes only CUDA tensors
except Exception:
    _ck_cuda = None
    _ck_cuda_available = False

try:
    _maybe_add_nvfp4_cu13_dll_dir()
    from lightx2v_kernel import gemm as _lx_gemm
    _lx_gemm_available = True
except Exception:
    _lx_gemm = None
    _lx_gemm_available = False

_NVFP4_QTYPE_NAME = "nvfp4"
if _NVFP4_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_NVFP4_QTYPE_NAME] = _quanto_qtype(
        _NVFP4_QTYPE_NAME,
        is_floating_point=True,
        bits=4,
        dtype=torch.uint8,
        qmin=-6.0,
        qmax=6.0,
    )
_NVFP4_QTYPE = _quanto_qtypes[_NVFP4_QTYPE_NAME]
HANDLER_PRIORITY = 1

_NVFP4_LAYOUT_LEGACY = "legacy"
_NVFP4_LAYOUT_TENSORCORE = "tensorcore"

_NVFP4_BACKEND_AUTO = "auto"
_NVFP4_BACKEND_COMFY = "comfy"
_NVFP4_BACKEND_LIGHTX2V = "lightx2v"
_NVFP4_BACKEND_TRITON = "triton"  # weight-only Triton kernels for every product (GPUs without FP4 tensor cores)

_NVFP4_KERNEL_LOGGED = False
_NVFP4_FALLBACK_LOGGED = False
_NVFP4_TRITON_LOGGED = False
_TRITON_W4A16 = None
_NVFP4_LOAD_LOGGED = False
_NVFP4_KERNEL_AVAILABLE = False
_NVFP4_KERNEL_CHECKED = False
_NVFP4_KERNEL_BACKEND = None
_NVFP4_AVAILABLE_BACKENDS = ()
_NVFP4_LOGGED_BACKENDS = set()
_NVFP4_ACT_SCALE_CACHE = {}

_NVFP4_SPLIT_FIELDS = {
    "weight": 0,
    "bias": 0,
    "weight_scale": 0,
    "weight_scale_2": 0,
    "pre_quant_scale": 0,
    "input_scale": 0,
    "input_global_scale": 0,
    "alpha": 0,
    "input_absmax": 0,
    "weight_global_scale": 0,
    "output_scale": 0,
}

_NVFP4_BACKEND_RAW = os.environ.get("WGP_NVFP4_BACKEND", _NVFP4_BACKEND_AUTO)

def _normalize_nvfp4_backend(name):
    if name is None:
        return _NVFP4_BACKEND_AUTO
    norm = str(name).strip().lower()
    if norm in ("", "auto", "default"):
        return _NVFP4_BACKEND_AUTO
    if norm in ("comfy", "comfy-kitchen", "comfy_kitchen", "ck"):
        return _NVFP4_BACKEND_COMFY
    if norm in ("lightx2v", "lightx2v_kernel", "lightx2v-kernel", "lx"):
        return _NVFP4_BACKEND_LIGHTX2V
    if norm in ("triton", "w4a16"):
        return _NVFP4_BACKEND_TRITON
    if norm in ("off", "none", "fallback", "disable", "disabled", "pytorch"):
        return "fallback"
    return norm

# Resolve the effective backend selection (env var WGP_NVFP4_BACKEND or auto).
_NVFP4_BACKEND = _normalize_nvfp4_backend(_NVFP4_BACKEND_RAW)


def _split_or_share_nvfp4_scale(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    total = sum(split_sizes)
    if src.numel() == 1:
        return [src] * len(split_sizes)
    if src.dim() > dim and src.size(dim) == total:
        return torch.split(src, split_sizes, dim=dim)
    if src.ndim > 1 and src.size(1) == total:
        return torch.split(src, split_sizes, dim=1)
    return [src] * len(split_sizes)


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload
    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields=dict(_NVFP4_SPLIT_FIELDS),
        split_handlers={
            "weight_scale": _split_or_share_nvfp4_scale,
            "weight_scale_2": _split_or_share_nvfp4_scale,
            "pre_quant_scale": _split_or_share_nvfp4_scale,
            "input_scale": _split_or_share_nvfp4_scale,
            "input_global_scale": _split_or_share_nvfp4_scale,
            "alpha": _split_or_share_nvfp4_scale,
            "input_absmax": _split_or_share_nvfp4_scale,
            "weight_global_scale": _split_or_share_nvfp4_scale,
            "output_scale": _split_or_share_nvfp4_scale,
        },
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )


_NVFP4_BACKEND = _normalize_nvfp4_backend(_NVFP4_BACKEND)


def _nvfp4_backend_candidates():
    if _NVFP4_BACKEND == _NVFP4_BACKEND_AUTO:
        return [_NVFP4_BACKEND_COMFY, _NVFP4_BACKEND_LIGHTX2V]
    if _NVFP4_BACKEND in (_NVFP4_BACKEND_COMFY, _NVFP4_BACKEND_LIGHTX2V):
        return [_NVFP4_BACKEND]
    return []


def _nvfp4_backend_label(backend):
    if backend == _NVFP4_BACKEND_LIGHTX2V:
        return "lightx2v"
    if backend == _NVFP4_BACKEND_COMFY:
        return "comfy-kitchen"
    return backend


def _nvfp4_lightx2v_device_ok(device):
    force = os.environ.get("WGP_NVFP4_LIGHTX2V_FORCE", "").strip().lower()
    if force in ("1", "true", "yes", "y"):
        return True
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    return props.major >= 12


def set_nvfp4_backend(name):
    global _NVFP4_BACKEND, _NVFP4_KERNEL_CHECKED, _NVFP4_KERNEL_AVAILABLE, _NVFP4_KERNEL_BACKEND
    global _NVFP4_KERNEL_LOGGED, _NVFP4_LOAD_LOGGED
    global _NVFP4_AVAILABLE_BACKENDS
    _NVFP4_BACKEND = _normalize_nvfp4_backend(name)
    _NVFP4_KERNEL_CHECKED = False
    _NVFP4_KERNEL_AVAILABLE = False
    _NVFP4_KERNEL_BACKEND = None
    _NVFP4_AVAILABLE_BACKENDS = ()
    _NVFP4_LOGGED_BACKENDS.clear()
    _NVFP4_KERNEL_LOGGED = False
    _NVFP4_LOAD_LOGGED = False


def _nvfp4_note_kernel(backend):
    global _NVFP4_KERNEL_LOGGED
    if backend not in _NVFP4_LOGGED_BACKENDS:
        label = _nvfp4_backend_label(backend)
        print(f"NVFP4: using {label} kernel")
        _NVFP4_LOGGED_BACKENDS.add(backend)
        _NVFP4_KERNEL_LOGGED = True


def _nvfp4_note_fallback():
    global _NVFP4_FALLBACK_LOGGED
    global _NVFP4_KERNEL_LOGGED
    if not _NVFP4_FALLBACK_LOGGED:
        if _NVFP4_KERNEL_LOGGED:
            print("NVFP4: linear fallback needed on some weights")
        else:
            print("NVFP4: linear fallback")
        _NVFP4_FALLBACK_LOGGED = True

def _nvfp4_note_reset():
    global _NVFP4_FALLBACK_LOGGED
    global _NVFP4_KERNEL_LOGGED
    global _NVFP4_LOAD_LOGGED
    _NVFP4_KERNEL_LOGGED = False
    _NVFP4_FALLBACK_LOGGED = False
    _NVFP4_LOAD_LOGGED = False
    _NVFP4_LOGGED_BACKENDS.clear()

def _nvfp4_note_load_backend():
    global _NVFP4_LOAD_LOGGED
    _init_nvfp4_kernel_support()  # when NVFP4 weights load, never during compilation: the probe keeps a kernel workspace in VRAM
    if _NVFP4_LOAD_LOGGED:
        return
    _NVFP4_LOAD_LOGGED = True
    if _NVFP4_KERNEL_AVAILABLE:
        label = ", ".join(_nvfp4_backend_label(b) for b in _NVFP4_AVAILABLE_BACKENDS)
        print(f"NVFP4: kernels available ({label}); selection follows weight packing and compatibility.")
    else:
        print("NVFP4: kernels unavailable; using fallback.")


def _check_nvfp4_kernel_support(device, backend):
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    if backend == _NVFP4_BACKEND_COMFY:
        if not _ck_cuda_available:
            return False
        if not hasattr(_ck_cuda, "scaled_mm_nvfp4"):
            return False
        if not hasattr(_ck_cuda, "quantize_nvfp4"):
            return False
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < (10, 0):
            return False
        # Exercise the installed binary and cuBLAS, not just its Python exports.
        with torch.inference_mode():
            scale = torch.ones((), device=device, dtype=torch.float32)
            for dtype in (torch.float16, torch.bfloat16):
                x = torch.full((16, 128), 6.0, device=device, dtype=dtype)
                packed, blocks = _ck_cuda.quantize_nvfp4(x, scale)
                out = _ck_cuda.scaled_mm_nvfp4(
                    packed, packed, scale, scale, blocks, blocks, out_dtype=dtype)
                if out.shape != (16, 16) or out.dtype != dtype or not (out == 4608).all().item():
                    return False
        return True
    if backend == _NVFP4_BACKEND_LIGHTX2V:
        if not _lx_gemm_available:
            return False
        if not _nvfp4_lightx2v_device_ok(device):
            return False
        if not (hasattr(torch.ops, "lightx2v_kernel") and hasattr(torch.ops.lightx2v_kernel, "cutlass_scaled_nvfp4_mm_sm120")):
            return False
        if not hasattr(torch.ops.lightx2v_kernel, "scaled_nvfp4_quant_sm120"):
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (12, 0)
    return False


def _init_nvfp4_kernel_support():
    global _NVFP4_KERNEL_AVAILABLE, _NVFP4_KERNEL_CHECKED, _NVFP4_KERNEL_BACKEND
    global _NVFP4_AVAILABLE_BACKENDS
    if _NVFP4_KERNEL_CHECKED:
        return
    if torch.compiler.is_compiling():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    _NVFP4_KERNEL_CHECKED = True
    _NVFP4_KERNEL_AVAILABLE = False
    _NVFP4_KERNEL_BACKEND = None
    available = []
    _NVFP4_AVAILABLE_BACKENDS = ()
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    for backend in _nvfp4_backend_candidates():
        try:
            if _check_nvfp4_kernel_support(device, backend):
                available.append(backend)
        except Exception:
            continue
    _NVFP4_AVAILABLE_BACKENDS = tuple(available)
    _NVFP4_KERNEL_AVAILABLE = bool(available)
    _NVFP4_KERNEL_BACKEND = available[0] if available else None


def _supports_nvfp4_kernel(device):
    if device.type != "cuda":
        return False
    if not _NVFP4_KERNEL_CHECKED:
        _init_nvfp4_kernel_support()
    return _NVFP4_KERNEL_AVAILABLE


def _is_fake_tensor(tensor):
    if isinstance(tensor, _TorchFakeTensor):
        return True
    if isinstance(tensor, _TorchFunctionalTensor):
        # During torch.compile pregraph (aot_autograd) fake tensors arrive
        # wrapped in a FunctionalTensor; unwrap to the base tensor.
        try:
            return _is_fake_tensor(tensor.from_functional())
        except Exception:
            elem = getattr(tensor, "elem", None)
            if torch.is_tensor(elem):
                return _is_fake_tensor(elem)
    return False


def _nvfp4_layout(weight):
    return getattr(weight, "_layout", _NVFP4_LAYOUT_LEGACY)


# ---------------------------------------------------------------------------
# Dispatch telemetry: lightweight counters so we can see, after a run, which
# NVFP4 linears took the CUDA kernel path vs the CPU fallback, and (when
# WGP_NVFP4_DEBUG=1) why a kernel was rejected.
# ---------------------------------------------------------------------------
_NVFP4_DISPATCH_STATS = {
    "kernel_allowed": 0,
    "kernel_denied": {},
    "linear_kernel": 0,
    "linear_fallback": 0,
    "linear_fake": 0,
}
_NVFP4_DEBUG = os.environ.get("WGP_NVFP4_DEBUG", "").strip().lower() not in ("", "0", "off", "false")
_NVFP4_DEBUG_PRINTED = 0


def _nvfp4_stat(kind, reason=None):
    if reason is None:
        _NVFP4_DISPATCH_STATS[kind] = _NVFP4_DISPATCH_STATS.get(kind, 0) + 1
        return
    denied = _NVFP4_DISPATCH_STATS["kernel_denied"]
    denied[reason] = denied.get(reason, 0) + 1


def _nvfp4_debug_line(context, input, weight, extra=""):
    global _NVFP4_DEBUG_PRINTED
    if not _NVFP4_DEBUG or _NVFP4_DEBUG_PRINTED >= 30:
        return
    _NVFP4_DEBUG_PRINTED += 1
    scale = getattr(weight, "_scale", None)
    igs = getattr(weight, "_input_global_scale", None)
    try:
        shape = tuple(input.shape)
    except Exception:
        shape = "?"
    scale_dev = scale.device if torch.is_tensor(scale) else type(scale).__name__
    igs_dev = igs.device if torch.is_tensor(igs) else type(igs).__name__
    print(
        f"[NVFP4DBG] {context} in={type(input).__name__} dev={getattr(input, 'device', '?')} "
        f"shape={shape} scale_dev={scale_dev} igs_dev={igs_dev} {extra}",
        flush=True,
    )


def _nvfp4_select_backend(input, weight):
    layout = _nvfp4_layout(weight)
    order = ((_NVFP4_BACKEND_LIGHTX2V, _NVFP4_BACKEND_COMFY) if layout == _NVFP4_LAYOUT_LEGACY
             else (_NVFP4_BACKEND_COMFY, _NVFP4_BACKEND_LIGHTX2V))
    k, n = input.shape[-1], weight.size(0)
    for backend in order:
        if backend not in _NVFP4_AVAILABLE_BACKENDS:
            continue
        if backend == _NVFP4_BACKEND_LIGHTX2V:
            if k % 32 == 0 and n % 32 == 0:
                return backend
        elif k % (64 if layout == _NVFP4_LAYOUT_LEGACY else 32) == 0 and n % 8 == 0:
            return backend
    return None


def nvfp4_dispatch_stats():
    return dict(_NVFP4_DISPATCH_STATS)


def nvfp4_reset_dispatch_stats():
    _NVFP4_DISPATCH_STATS.update(
        {"kernel_allowed": 0, "kernel_denied": {}, "linear_kernel": 0, "linear_fallback": 0, "linear_fake": 0}
    )


def _nvfp4_can_use_kernel(input, weight):
    reason = None
    if not torch.is_tensor(input):
        reason = "input_not_tensor"
    elif not getattr(weight, "_allow_kernel", True):
        reason = "allow_kernel_false"
    elif not _supports_nvfp4_kernel(input.device):
        reason = f"no_kernel_support_{input.device.type}"
    else:
        backend = _nvfp4_select_backend(input, weight)
        if backend is None:
            reason = "backend_unavailable"
        else:
            layout = _nvfp4_layout(weight)
            if backend == _NVFP4_BACKEND_LIGHTX2V:
                if input.shape[-1] % 32 != 0:
                    reason = f"lx_in_dim_{input.shape[-1]}"
                elif weight.size(0) % 32 != 0:
                    reason = f"lx_out_dim_{weight.size(0)}"
            else:
                if layout == _NVFP4_LAYOUT_LEGACY:
                    if input.shape[-1] % 64 != 0:
                        reason = f"ck_in_dim_{input.shape[-1]}"
                else:
                    if input.shape[-1] % 16 != 0:
                        reason = f"ck_in_dim_{input.shape[-1]}"
                if reason is None and weight.size(0) % 8 != 0:
                    reason = f"ck_out_dim_{weight.size(0)}"
            if reason is None and weight._data.shape[1] * 2 != input.shape[-1]:
                reason = f"k_dim_mismatch_{weight._data.shape[1] * 2}_vs_{input.shape[-1]}"
            if reason is None and weight._block_size != 16:
                reason = f"block_size_{weight._block_size}"
            if reason is None and not (torch.is_tensor(weight._input_global_scale) and torch.is_tensor(weight._alpha)):
                reason = "missing_scales"
            if reason is None and getattr(weight._input_global_scale, "is_meta", False):
                reason = "input_global_scale_is_meta"
            if reason is None and not _is_fake_tensor(input):
                finite = weight.__dict__.get("_finite_input_scale")
                if finite is None:  # checked once per weight: a device sync on every call otherwise, which CUDA graphs cannot capture
                    try:
                        finite = bool(torch.isfinite(weight._input_global_scale).all())
                    except Exception as exc:
                        finite = False
                        reason = f"isfinite_error_{type(exc).__name__}"
                    weight._finite_input_scale = finite
                if finite is False and reason is None:
                    reason = "input_global_scale_nonfinite"
            if reason is None:
                _nvfp4_stat("kernel_allowed")
                return True
    if reason is not None:
        _nvfp4_stat("kernel_denied", reason)
        _nvfp4_debug_line("denied", input, weight, extra=f"reason={reason}")
    return False


def _nvfp4_get_act_scale(device):
    act_scale = _NVFP4_ACT_SCALE_CACHE.get(device)
    if act_scale is None:
        act_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        _NVFP4_ACT_SCALE_CACHE[device] = act_scale
    return act_scale


def _nvfp4_swap_nibbles(tensor):
    return ((tensor & 0x0F) << 4) | ((tensor & 0xF0) >> 4)


_ROW_SLICE_BYTES = 64 << 20  # scratch of the activation rows quantized and multiplied at a time: FP4 values and block scales
_ROW_SLICE_MIN = 16384  # rows per FP4 GEMM, which lose tensor core throughput on fewer rows


def _nvfp4_rows(x2d, out_shape, out_dtype, quantize, gemm):
    """gemm(quantize(x), out rows) (FP4 activations multiplied into their rows of the output) by slices of 128-row tiles when x2d
    spans two or more: the FP4 copy of x2d never exceeds one slice. Its scales are per block of 16 values of a row with a fixed
    tensor scale, so the slices quantize as the whole tensor does."""
    rows, k = x2d.shape
    step = max(_ROW_SLICE_MIN, _ROW_SLICE_BYTES // (k // 2 + k // 16) // 128 * 128)
    step = -(-rows // (max(1, rows // step) * 128)) * 128  # slices of even size, none below the minimum
    out = None
    for start in range(0, rows, step):
        prepared = quantize(x2d[start:start + step])
        if out is None:  # allocated once the temporaries of the first quantization are released
            out = torch.empty(out_shape, dtype=out_dtype, device=x2d.device)
        gemm(prepared, out[start:start + step])
        del prepared
    return out


def _kitchen_scaled_mm_nvfp4(out, qx, weight_fp4, qx_scale, weight_scale, alpha, bias):
    """comfy-kitchen 0.2.35 scaled_mm_nvfp4 (cuBLAS FP4 GEMM) written to out, rows of a larger output (the wrapper allocates its own)."""
    wrap = _ck_cuda._wrap_for_dlpack
    bias = torch.empty(0, device=qx.device, dtype=torch.float16) if bias is None else bias
    _ck_cuda._C.cublas_gemm_blockwise_fp4(wrap(weight_fp4), wrap(weight_scale.view(torch.uint8)), wrap(qx), wrap(qx_scale.view(torch.uint8)),
                                          wrap(out), _ck_cuda.DTYPE_TO_CODE[out.dtype], wrap(bias), wrap(_ck_cuda.get_cublas_workspace()),
                                          False, wrap(alpha), torch.cuda.current_stream(qx.device).cuda_stream)


def _nvfp4_linear_cuda_comfy(input, weight, bias=None):
    _nvfp4_note_kernel(_NVFP4_BACKEND_COMFY)
    x2d = input.reshape(-1, input.shape[-1])
    if not x2d.is_floating_point():
        x2d = x2d.to(torch.float16)
    orig_dtype = x2d.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        x2d = x2d.to(torch.float16)
        out_dtype = torch.float16
    else:
        out_dtype = orig_dtype
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    weight_fp4 = weight._data
    weight_scale = weight._scale
    input_scale = weight._input_global_scale
    alpha = weight._alpha
    layout = _nvfp4_layout(weight)
    device = x2d.device
    if weight_fp4.device != device:
        weight_fp4 = weight_fp4.to(device)
    if weight_scale.device != device:
        weight_scale = weight_scale.to(device)
    if input_scale.device != device:
        input_scale = input_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if bias is not None and torch.is_tensor(bias) and bias.dtype != out_dtype:
        bias = bias.to(out_dtype)
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        act_scale = input_scale.to(torch.float32)
        alpha = act_scale * alpha.to(torch.float32)
    else:
        alpha = alpha * input_scale
        act_scale = _nvfp4_get_act_scale(device)
        weight_fp4 = _nvfp4_swap_nibbles(weight_fp4)
    alpha = alpha.to(torch.float32).reshape(1)
    # cuBLAS takes the activations padded to 16 rows, written to the output rows padded the same way
    out = _nvfp4_rows(x2d, (-(-x2d.shape[0] // 16) * 16, weight.size(0)), out_dtype,
                      lambda x: _ck_cuda.quantize_nvfp4(x, act_scale, 0.0, x.shape[0] % 16 != 0),
                      lambda q, out: _kitchen_scaled_mm_nvfp4(out[:q[0].shape[0]], q[0], weight_fp4, q[1], weight_scale, alpha, bias))[:x2d.shape[0]]
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out.reshape(*input.shape[:-1], weight.size(0))


def _nvfp4_linear_cuda_lightx2v(input, weight, bias=None):
    _nvfp4_note_kernel(_NVFP4_BACKEND_LIGHTX2V)
    x2d = input.reshape(-1, input.shape[-1])
    if not x2d.is_floating_point():
        x2d = x2d.to(torch.float16)
    orig_dtype = x2d.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        x2d = x2d.to(torch.float16)
        out_dtype = torch.float16
    else:
        out_dtype = orig_dtype
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    weight_fp4 = weight._data
    weight_scale = weight._scale
    input_scale = weight._input_global_scale
    alpha = weight._alpha
    layout = _nvfp4_layout(weight)
    device = x2d.device
    if weight_fp4.device != device:
        weight_fp4 = weight_fp4.to(device)
    if weight_scale.device != device:
        weight_scale = weight_scale.to(device)
    if not weight_fp4.is_contiguous():
        weight_fp4 = weight_fp4.contiguous()
    if not weight_scale.is_contiguous():
        weight_scale = weight_scale.contiguous()
    if input_scale.device != device:
        input_scale = input_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if input_scale.dtype != torch.float32:
        input_scale = input_scale.to(torch.float32)
    if alpha.dtype != torch.float32:
        alpha = alpha.to(torch.float32)
    if bias is not None and torch.is_tensor(bias):
        if bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        if not bias.is_contiguous():
            bias = bias.contiguous()
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        # lightx2v dequantizes A to x/input_scale (global-scaled) units and B
        # to W/input_global_scale_w, so the output alpha is weight_scale_2 *
        # input_scale (== 1/(gs_x * gs_w)).
        quant_scale = torch.reciprocal(torch.clamp(input_scale, min=1e-8))
        alpha = alpha * input_scale
    else:
        quant_scale = input_scale

    def quantize(x):
        qx, qx_scale = _lx_gemm.scaled_nvfp4_quant(x, quant_scale)
        if layout == _NVFP4_LAYOUT_TENSORCORE:  # nibbles of the uint8 activations swapped in place: one temporary
            low = qx & 0x0F
            qx.bitwise_right_shift_(4).bitwise_or_(low.bitwise_left_shift_(4))
            del low
        return qx.contiguous(), qx_scale.contiguous()

    # the op behind lightx2v's cutlass_scaled_nvfp4_mm, which allocates its own BF16 output
    out = _nvfp4_rows(x2d, (x2d.shape[0], weight.size(0)), torch.bfloat16, quantize,
                      lambda q, out: torch.ops.lightx2v_kernel.cutlass_scaled_nvfp4_mm_sm120.default(out, q[0], weight_fp4, q[1], weight_scale, alpha, bias))
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out.reshape(*input.shape[:-1], weight.size(0))


def _nvfp4_linear_cuda(input, weight, bias=None):
    if _nvfp4_select_backend(input, weight) == _NVFP4_BACKEND_LIGHTX2V:
        return _nvfp4_linear_cuda_lightx2v(input, weight, bias=bias)
    return _nvfp4_linear_cuda_comfy(input, weight, bias=bias)


def _nvfp4_fake_new_empty(input, weight):
    # Return an empty (fake) tensor matching the linear output shape without
    # touching any real tensor (LUT materialization would crash pregraph).
    base = input
    for _ in range(4):
        if not isinstance(base, _TorchFunctionalTensor):
            break
        try:
            base = base.from_functional()
        except Exception:
            elem = getattr(base, "elem", None)
            if torch.is_tensor(elem):
                base = elem
            else:
                break
    if torch.is_tensor(base):
        return base.new_empty((*input.shape[:-1], weight.size(0)))
    return input.new_empty((*input.shape[:-1], weight.size(0)))


def _triton_w4a16():
    """The weight-only Triton kernels (GPUs without FP4 tensor cores), None when Triton is unavailable or the backend disables them."""
    global _TRITON_W4A16
    if _TRITON_W4A16 is None:
        _TRITON_W4A16 = False
        if _NVFP4_BACKEND not in ("pytorch", "fallback"):
            try:
                from shared.kernels import nvfp4_w4a16_triton
                _TRITON_W4A16 = nvfp4_w4a16_triton
            except ImportError:
                pass
    return _TRITON_W4A16 or None


def _nvfp4_note_triton():
    global _NVFP4_TRITON_LOGGED
    if not _NVFP4_TRITON_LOGGED:
        _NVFP4_TRITON_LOGGED = True
        print("NVFP4: Triton W4A16 kernels (weight-only, no FP4 tensor cores on this GPU)")


_TRITON_W4A16_MAX_ROWS = 256  # measured crossover on an RTX 4090: above it, Kitchen's CUDA dequantization + cuBLAS BF16 is faster


def _prefer_triton_w4a16(input, weight):
    """The weight-only Triton kernel beats dequantize + matmul when the product is memory-bound (few rows), when Kitchen's CUDA
    dequantization is unavailable (the PyTorch expansion is slow) or for the legacy layout it does not expand."""
    if _NVFP4_BACKEND == _NVFP4_BACKEND_TRITON:
        return True
    rows = input.numel() // input.shape[-1]
    return rows <= _TRITON_W4A16_MAX_ROWS or not _ck_cuda_available or weight._layout != _NVFP4_LAYOUT_TENSORCORE


@torch.compiler.disable()
def _nvfp4_linear(input, weight, bias=None, op=None):
    if _nvfp4_can_use_kernel(input, weight):
        if _is_fake_tensor(input):
            _nvfp4_stat("linear_fake")
            return _nvfp4_fake_new_empty(input, weight)
        _nvfp4_stat("linear_kernel")
        _nvfp4_debug_line("kernel", input, weight, extra=f"backend={_NVFP4_KERNEL_BACKEND}")
        return _nvfp4_linear_cuda(input, weight, bias=bias)
    if _is_fake_tensor(input):
        _nvfp4_stat("linear_fake")
        return _nvfp4_fake_new_empty(input, weight)
    triton_w4a16 = _triton_w4a16() if torch.is_tensor(input) and weight._pre_quant_scale is None else None
    if (triton_w4a16 is not None and triton_w4a16.supported(weight, input) and weight._data.device == input.device
            and _prefer_triton_w4a16(input, weight)):
        _nvfp4_note_triton()
        return triton_w4a16.linear(input, weight, bias)
    _nvfp4_stat("linear_fallback")
    _nvfp4_debug_line("fallback", input, weight)
    _nvfp4_note_fallback()

    if torch.is_tensor(input):
        qweight = weight.dequantize(dtype=input.dtype, device=input.device)
        bias_arg = bias
        if bias_arg is not None and torch.is_tensor(bias_arg):
            if bias_arg.device != input.device or bias_arg.dtype != input.dtype:
                bias_arg = bias_arg.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, qweight, bias=bias_arg)

    return torch.nn.functional.linear(input, weight.dequantize(), bias=bias)


# ---------------------------------------------------------------------------
# torch.compile support: torch.ops.wgp_nvfp4.gemm
#
# Under torch.compile the NVFP4 weight arrives as a (Functional)Tensor subclass
# during graph tracing. Two naive approaches both break:
#   * a plain @torch.compiler.disable() graph break: inductor can bake the
#     trace-time *empty* tensor into the compiled graph so the kernel never
#     runs at runtime (black video); and
#   * a custom op that takes the weight *subclass* as an argument: the
#     dispatcher strips the subclass for a torch.library custom op, so the
#     quantized state stored as Python attributes (_data/_scale/_alpha/...)
#     is lost inside the op.
#
# The robust pattern is a custom op whose arguments are all *plain* tensors
# (the activation plus the weight's stored sub-tensors). At trace time the
# registered fake impl only provides the output shape, so inductor records a
# real opaque node; at runtime it invokes the real impl, which runs the actual
# NVFP4 kernel (or the dequant fallback) on real data.
# ---------------------------------------------------------------------------
_NVFP4_LAYOUT_ID = {_NVFP4_LAYOUT_LEGACY: 0, _NVFP4_LAYOUT_TENSORCORE: 1}
_NVFP4_LAYOUT_FROM_ID = {v: k for k, v in _NVFP4_LAYOUT_ID.items()}
_NVFP4_BACKEND_ID = {_NVFP4_BACKEND_LIGHTX2V: 0, _NVFP4_BACKEND_COMFY: 1}
_NVFP4_BACKEND_FROM_ID = {v: k for k, v in _NVFP4_BACKEND_ID.items()}


class _Nvfp4PlainWeight:
    """Minimal standin exposing the attributes the CUDA kernel helpers read
    from an NVFP4WeightTensor, backed by plain tensors (so it survives the
    torch.library custom-op dispatch which strips tensor subclasses)."""

    __slots__ = ("_data", "_scale", "_input_global_scale", "_alpha", "_layout", "_out")

    def __init__(self, data, scale, input_global_scale, alpha, layout, out_features):
        self._data = data
        self._scale = scale
        self._input_global_scale = input_global_scale
        self._alpha = alpha
        self._layout = layout
        self._out = out_features

    def size(self, dim=None):
        if dim is None:
            return (self._out, self._data.shape[1] * 2)
        return self._out if dim == 0 else self._data.shape[1] * 2


def _nvfp4_gemm_real(input, weight_fp4, weight_scale, input_scale, alpha, bias, layout_id, backend_id, use_kernel):
    layout = _NVFP4_LAYOUT_FROM_ID.get(layout_id, _NVFP4_LAYOUT_LEGACY)
    backend = _NVFP4_BACKEND_FROM_ID.get(backend_id, _NVFP4_BACKEND_LIGHTX2V)
    if use_kernel and backend in (_NVFP4_BACKEND_LIGHTX2V, _NVFP4_BACKEND_COMFY):
        standin = _Nvfp4PlainWeight(
            weight_fp4, weight_scale, input_scale, alpha, layout, weight_fp4.shape[0]
        )
        # Count this real kernel execution (the graph-break/fallback dispatcher
        # _nvfp4_linear does the equivalent when the custom op is unavailable).
        _nvfp4_stat("linear_kernel")
        _nvfp4_debug_line(
            "gemm_real", input, standin,
            extra=(
                f"backend={backend} in={tuple(input.shape)} in_dev={input.device} "
                f"w_dev={weight_fp4.device} layout={layout}"
            ),
        )
        if backend == _NVFP4_BACKEND_LIGHTX2V:
            return _nvfp4_linear_cuda_lightx2v(input, standin, bias=bias)
        return _nvfp4_linear_cuda_comfy(input, standin, bias=bias)
    # No kernel available (e.g. WGP_NVFP4_BACKEND=fallback): dequantize the
    # plain sub-tensors and run a regular linear. Doing this inside the op
    # (instead of a @torch.compiler.disable() graph break) keeps the compiled
    # graph correct: a graph break would let inductor bake the trace-time
    # empty tensor in and never execute the real computation at runtime.
    _nvfp4_note_fallback()
    _nvfp4_stat("linear_fallback")
    _nvfp4_debug_line("gemm_dequant", input, None)
    qweight = _dequantize_nvfp4_weight(
        weight_u8=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale=input_scale,
        alpha=alpha,
        dtype=input.dtype,
        device=input.device,
        block_size=16,
        layout=layout,
    )
    bias_arg = bias
    if bias_arg is not None and torch.is_tensor(bias_arg):
        if bias_arg.device != input.device or bias_arg.dtype != input.dtype:
            bias_arg = bias_arg.to(device=input.device, dtype=input.dtype)
    return torch.nn.functional.linear(input, qweight, bias=bias_arg)


def _nvfp4_gemm_fake(input, weight_fp4, weight_scale, input_scale, alpha, bias, layout_id, backend_id, use_kernel):
    return input.new_empty(*input.shape[:-1], weight_fp4.shape[0])


_NVFP4_HAS_CUSTOM_OP = False
try:
    # NOTE: in torch 2.9 `fn` is a positional-only argument of custom_op();
    # passing it as a keyword raises TypeError (and a silent registration
    # failure would leave the broken graph-break path active under compile).
    _wgp_nvfp4_gemm_op = torch.library.custom_op(
        "wgp_nvfp4::gemm",
        _nvfp4_gemm_real,
        mutates_args=(),
        device_types=("cpu", "cuda"),
        schema=(
            "(Tensor input, Tensor weight_fp4, Tensor weight_scale, "
            "Tensor input_scale, Tensor alpha, Tensor? bias, "
            "int layout_id, int backend_id, bool use_kernel) -> Tensor"
        ),
    )
    _wgp_nvfp4_gemm_op.register_fake(_nvfp4_gemm_fake)
    _NVFP4_HAS_CUSTOM_OP = True
    if os.environ.get("WGP_NVFP4_DEBUG", "").strip().lower() not in ("", "0", "off", "false"):
        print("[NVFP4] custom op torch.ops.wgp_nvfp4.gemm registered", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    _wgp_nvfp4_gemm_op = None


def _is_float8_dtype(dtype):
    return "float8" in str(dtype).lower() or "f8" in str(dtype).lower()

_FP4_LUT_BASE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_LUT_CACHE = {}
_FP4_BYTE_LUT_CACHE = {}


_DEQUANT_SLICE_BYTES = 32 << 20  # temporaries of the portable dequantization for one slice of rows


def _get_fp4_lut(device, dtype):
    key = (device, dtype)
    lut = _FP4_LUT_CACHE.get(key)
    if lut is None:
        lut = _FP4_LUT_BASE.to(device=device, dtype=dtype)
        _FP4_LUT_CACHE[key] = lut
    return lut


def _get_fp4_byte_lut(device, dtype):
    key = (device, dtype)
    byte_lut = _FP4_BYTE_LUT_CACHE.get(key)
    if byte_lut is None:
        lut16 = _get_fp4_lut(device, dtype)
        b = torch.arange(256, device=device, dtype=torch.int32)
        byte_lut = torch.empty((256, 2), device=device, dtype=dtype)
        byte_lut[:, 0] = lut16[b & 0x0F]
        byte_lut[:, 1] = lut16[b >> 4]
        _FP4_BYTE_LUT_CACHE[key] = byte_lut
    return byte_lut


def _deswizzle_nvfp4_scale(scale, in_features, block_size=16, dtype=None):
    k_groups = in_features // block_size
    if scale.shape[1] < k_groups:
        raise RuntimeError(
            f"NVFP4 scale shape mismatch: expected at least {k_groups} groups, got {scale.shape[1]}"
        )
    # Deswizzle on the stored (padded) width first, then truncate the logical
    # result: the 128x64 swizzle tiles are defined over the full stored grid,
    # so truncating before the permute would corrupt the mapping (or crash the
    # reshape) when a file stores more scale columns than in_features // 16.
    m, stored_cols = scale.shape
    k_tiles = (stored_cols + 3) // 4
    m_tiles = (m + 128 - 1) // 128
    tmp = scale if dtype is None else scale.to(dtype)
    tmp = tmp.reshape(1, m_tiles, k_tiles, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    out = tmp.reshape(m_tiles * 128, k_tiles * 4)
    return out[:m, :k_groups]


def _dequantize_nvfp4_weight(
    weight_u8,
    weight_scale,
    input_global_scale,
    alpha,
    dtype,
    device,
    block_size=16,
    layout=_NVFP4_LAYOUT_LEGACY,
):
    if weight_u8.device != device:
        weight_u8 = weight_u8.to(device)
    scale = weight_scale if weight_scale.device == device else weight_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if input_global_scale.device != device:
        input_global_scale = input_global_scale.to(device)
    # Kitchen's dequantize_nvfp4 contract: contiguous input, rows and unpacked columns in multiples of 16, FP32/FP16/BF16 output
    if (layout == _NVFP4_LAYOUT_TENSORCORE and device.type == "cuda" and _ck_cuda_available and weight_u8.shape[0] % 16 == 0
            and weight_u8.shape[1] % 8 == 0 and weight_u8.is_contiguous() and dtype in (torch.float32, torch.float16, torch.bfloat16)):
        return _ck_cuda.dequantize_nvfp4(weight_u8, alpha.to(torch.float32), scale, output_type=dtype)

    m, k_bytes = weight_u8.shape
    if scale.shape[0] < m:
        raise RuntimeError(f"NVFP4 scale row mismatch: expected at least {m} rows, got {scale.shape[0]}")
    # FP4 value * block scale is exact in FP32 (FP64 for an FP64 output); the per-tensor factor rounds it once more, then the output dtype
    compute = torch.promote_types(dtype, torch.float32)
    byte_lut = _get_fp4_byte_lut(device, compute)
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        byte_lut = byte_lut.flip(1)  # tensorcore packing stores the first value of a byte in its high nibble
        factor = alpha.to(compute)
    else:
        factor = alpha.to(compute) * input_global_scale.to(compute)
    block_scale = _deswizzle_nvfp4_scale(scale, k_bytes * 2, block_size=block_size)[:m]
    out = torch.empty((m, k_bytes * 2), dtype=dtype, device=device)
    step = max(1, _DEQUANT_SLICE_BYTES // (k_bytes * (4 + 2 * compute.itemsize)))  # int32 indices and two compute values per packed byte
    for start in range(0, m, step):
        rows = min(step, m - start)
        values = torch.nn.functional.embedding(weight_u8[start:start + rows].to(torch.int32), byte_lut).view(rows, -1, block_size)
        values.mul_(block_scale[start:start + rows].to(compute).unsqueeze(-1))
        torch.mul(values.view(rows, -1), factor, out=out[start:start + rows])
        del values
    return out


def _collect_nvfp4_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if tensor.dtype != torch.uint8:
            continue
        base = key[:-7]
        scale_key = base + ".weight_scale"
        if scale_key not in state_dict:
            continue
        if not _is_float8_dtype(state_dict[scale_key].dtype):
            continue

        weight_scale_2_key = base + ".weight_scale_2"
        input_scale_key = base + ".input_scale"
        if weight_scale_2_key in state_dict:
            specs.append(
                {
                    "name": base,
                    "weight": tensor,
                    "weight_scale": state_dict[scale_key],
                    "weight_scale_2": state_dict[weight_scale_2_key],
                    "pre_quant_scale": state_dict.get(base + ".pre_quant_scale", None),
                    "input_scale": state_dict.get(input_scale_key, None),
                    "bias": state_dict.get(base + ".bias", None),
                    "layout": _NVFP4_LAYOUT_TENSORCORE,
                }
            )
            continue

        input_global_key = base + ".input_global_scale"
        alpha_key = base + ".alpha"
        input_absmax_key = base + ".input_absmax"
        weight_global_scale_key = base + ".weight_global_scale"
        if input_global_key not in state_dict or alpha_key not in state_dict:
            if input_absmax_key not in state_dict or weight_global_scale_key not in state_dict:
                continue
            input_absmax = state_dict[input_absmax_key]
            weight_global_scale = state_dict[weight_global_scale_key]
            input_global_scale = (2688.0 / input_absmax).to(torch.float32)
            alpha = 1.0 / (input_global_scale * weight_global_scale.to(torch.float32))
        else:
            input_global_scale = state_dict[input_global_key]
            alpha = state_dict[alpha_key]
        specs.append(
            {
                "name": base,
                "weight": tensor,
                "weight_scale": state_dict[scale_key],
                "pre_quant_scale": state_dict.get(base + ".pre_quant_scale", None),
                "input_global_scale": input_global_scale,
                "alpha": alpha,
                "bias": state_dict.get(base + ".bias", None),
                "layout": _NVFP4_LAYOUT_LEGACY,
            }
        )
    return specs


def detect_nvfp4_state_dict(state_dict):
    return len(_collect_nvfp4_specs(state_dict)) > 0


def describe_nvfp4_state_dict(state_dict, max_names=8):
    specs = _collect_nvfp4_specs(state_dict)
    names = [spec["name"] for spec in specs]
    return {"count": len(names), "names": names[:max_names]}


def convert_nvfp4_to_quanto(state_dict, default_dtype=None, verboseLevel=1):
    specs = _collect_nvfp4_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    _nvfp4_note_load_backend()
    quant_map = {}
    for spec in specs:
        qcfg = {"weights": "nvfp4", "activations": "none"}
        quant_map[spec["name"]] = qcfg
        quant_map[spec["name"] + ".weight"] = qcfg
    return {"state_dict": state_dict, "quant_map": quant_map}


def cpu_compute_supported(tensor):
    """Whether linears with this weight compute on the CPU in its storage when MMGP leaves their block there (cpuCompute)."""
    from shared.kernels import cpu_linear
    return isinstance(tensor, NVFP4WeightTensor) and cpu_linear.supported(tensor)


def detect(state_dict, verboseLevel=1):
    matched = detect_nvfp4_state_dict(state_dict)
    details = describe_nvfp4_state_dict(state_dict) if matched else {}
    return {"matched": matched, "kind": "nvfp4" if matched else "none", "details": details}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    _nvfp4_note_reset()    
    return convert_nvfp4_to_quanto(state_dict, default_dtype=default_dtype, verboseLevel=verboseLevel)


class _NVFP4RowProjection:
    @staticmethod
    def supports(module, x):
        weight = getattr(module, "weight", None)
        return isinstance(weight, NVFP4WeightTensor) and not _is_fake_tensor(x) and x.dtype in (torch.float16, torch.bfloat16)

    @staticmethod
    def key(module):
        weight = module.weight
        backend = _nvfp4_select_backend(weight, weight) if _nvfp4_can_use_kernel(weight, weight) else None
        scale = weight._input_global_scale if backend != _NVFP4_BACKEND_COMFY or weight._layout == _NVFP4_LAYOUT_TENSORCORE else None
        return "nvfp4_rows", backend, weight._layout, weight.dtype, scale, weight._pre_quant_scale

    @staticmethod
    def prepare(module, x):
        weight = module.weight
        if weight._pre_quant_scale is not None:
            x = x * weight._pre_quant_scale.to(x.dtype)
        backend = _NVFP4RowProjection.key(module)[1]
        if backend is None:
            return x
        x = x.contiguous()
        if backend == _NVFP4_BACKEND_COMFY:
            scale = weight._input_global_scale.float() if weight._layout == _NVFP4_LAYOUT_TENSORCORE else _nvfp4_get_act_scale(x.device)
            q, scales = _ck_cuda.quantize_nvfp4(x, scale, 0.0, x.shape[0] % 16 != 0)
        else:
            scale = weight._input_global_scale.float()
            if weight._layout == _NVFP4_LAYOUT_TENSORCORE:
                scale = torch.reciprocal(torch.clamp(scale, min=1e-8))
            q, scales = _lx_gemm.scaled_nvfp4_quant(x, scale)
            if weight._layout == _NVFP4_LAYOUT_TENSORCORE:
                q = _nvfp4_swap_nibbles(q)
        return q, scales, x.shape[0], x.dtype

    @staticmethod
    def rows(module, prepared, start, stop):
        weight = module.weight
        # Swizzled block scales store contiguous groups of 128 output rows.
        first, last = start // 128 * 128, min((stop + 127) // 128 * 128, weight.shape[0])
        data = weight._data[first:last]
        scales = weight._scale[first:(last + 127) // 128 * 128]
        bias = None if module.bias is None else module.bias[first:last]
        backend = _NVFP4RowProjection.key(module)[1]
        if backend is None:
            triton_w4a16 = _triton_w4a16() if weight._pre_quant_scale is None else None
            if (triton_w4a16 is not None and triton_w4a16.supported(weight, prepared) and (last - first) % 128 == 0
                    and _prefer_triton_w4a16(prepared, weight)):
                _nvfp4_note_triton()
                out = triton_w4a16.linear(prepared, triton_w4a16.RowSlice(weight, first, last), bias)
            else:
                dense = _dequantize_nvfp4_weight(data, scales, weight._input_global_scale, weight._alpha, prepared.dtype, prepared.device, layout=weight._layout)
                out = torch.nn.functional.linear(prepared, dense, bias)
        else:
            q, qs, rows, dtype = prepared
            input_scale, alpha = weight._input_global_scale, weight._alpha
            if backend == _NVFP4_BACKEND_COMFY:
                if weight._layout == _NVFP4_LAYOUT_TENSORCORE:
                    out = _ck_cuda.scaled_mm_nvfp4(q, data, input_scale.float(), alpha.float(), qs, scales,
                                                  bias=None if bias is None else bias.to(dtype), out_dtype=dtype)
                else:
                    out = _ck_cuda.scaled_mm_nvfp4(q, _nvfp4_swap_nibbles(data), _nvfp4_get_act_scale(q.device), input_scale,
                                                  qs, scales, bias=None if bias is None else bias.to(dtype), out_dtype=dtype,
                                                  alpha=(alpha * input_scale).float())
            else:
                alpha = alpha.float()
                if weight._layout == _NVFP4_LAYOUT_TENSORCORE:
                    alpha = alpha * input_scale.float()
                out = _lx_gemm.cutlass_scaled_nvfp4_mm(q.contiguous(), data.contiguous(), qs.contiguous(), scales.contiguous(),
                                                       alpha=alpha, bias=None if bias is None else bias.to(torch.bfloat16).contiguous())
            out = out[:rows].to(dtype)
        return out[:, start - first:stop - first]


from mmgp import offload

_NVFP4_ROW_PROJECTION = _NVFP4RowProjection()
offload.register_row_projection(_NVFP4_ROW_PROJECTION)


class Int8TensorwiseEmbedding(torch.nn.Embedding):
    def __init__(self, module, output_dtype):
        super().__init__(module.num_embeddings, module.embedding_dim, module.padding_idx, module.max_norm, module.norm_type,
                         module.scale_grad_by_freq, module.sparse, device=module.weight.device, dtype=module.weight.dtype)
        self.register_buffer("weight_scale", torch.empty((module.num_embeddings, 1), device=module.weight.device, dtype=torch.float32))
        self.output_dtype = output_dtype
        self._lock_dtype = torch.int8
        self.requires_grad_(False)

    def forward(self, input):
        weight = torch.nn.functional.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                                                self.scale_grad_by_freq, self.sparse).to(self.output_dtype)
        scale = torch.nn.functional.embedding(input, self.weight_scale).to(self.output_dtype)
        return weight.mul_(scale)


def _is_int8_tensorwise(config):
    if not torch.is_tensor(config) or config.dtype != torch.uint8 or config.numel() > 256:
        return False
    try:
        return ast.literal_eval(bytes(config.tolist()).decode()).get("format") == "int8_tensorwise"
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return False


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    for key in [key for key in state_dict if key.endswith(".comfy_quant") and _is_int8_tensorwise(state_dict[key])]:
        name = key.removesuffix(".comfy_quant")
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Embedding):
            raise ValueError(f"Unsupported tensorwise INT8 module: {name}")
        scale = state_dict.get(name + ".weight_scale")
        if scale is None or tuple(scale.shape) != (module.num_embeddings, 1):
            raise ValueError(f"Invalid tensorwise INT8 embedding scale: {name}")
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, Int8TensorwiseEmbedding(module, default_dtype))
        state_dict.pop(key)
    return quantization_map, []


def _nvfp4_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(NVFP4WeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class NVFP4WeightTensor(QTensor):
    @staticmethod
    def create(
        weight_u8,
        weight_scale,
        size,
        stride,
        dtype,
        input_global_scale=None,
        alpha=None,
        input_scale=None,
        weight_scale_2=None,
        pre_quant_scale=None,
        device=None,
        requires_grad=False,
        layout=_NVFP4_LAYOUT_LEGACY,
        allow_kernel=True,
    ):
        if input_global_scale is None and input_scale is not None:
            input_global_scale = input_scale
        if alpha is None and weight_scale_2 is not None:
            alpha = weight_scale_2
        if layout == _NVFP4_LAYOUT_LEGACY and (weight_scale_2 is not None or input_scale is not None):
            layout = _NVFP4_LAYOUT_TENSORCORE
        if input_global_scale is None or alpha is None:
            raise ValueError("NVFP4WeightTensor.create requires input_global_scale/alpha or input_scale/weight_scale_2")
        if torch.is_tensor(input_global_scale):
            try:
                if not torch.isfinite(input_global_scale).all():
                    allow_kernel = False
            except Exception:
                allow_kernel = False
        device = weight_u8.device if device is None else device
        if weight_u8.device != device:
            weight_u8 = weight_u8.to(device)
        if weight_scale.device != device:
            weight_scale = weight_scale.to(device)
        if input_global_scale.device != device:
            input_global_scale = input_global_scale.to(device)
        if alpha.device != device:
            alpha = alpha.to(device)
        if pre_quant_scale is not None and pre_quant_scale.device != device:
            pre_quant_scale = pre_quant_scale.to(device)
        return NVFP4WeightTensor(
            qtype=_NVFP4_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            weight_u8=weight_u8,
            weight_scale=weight_scale,
            input_global_scale=input_global_scale,
            alpha=alpha,
            pre_quant_scale=pre_quant_scale,
            allow_kernel=allow_kernel,
            dtype=dtype,
            requires_grad=requires_grad,
            layout=layout,
        )

    @staticmethod
    def __new__(
        cls,
        qtype,
        axis,
        size,
        stride,
        weight_u8,
        weight_scale,
        input_global_scale,
        alpha,
        dtype,
        pre_quant_scale=None,
        allow_kernel=True,
        requires_grad=False,
        layout=_NVFP4_LAYOUT_LEGACY,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=weight_u8.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        qtype,
        axis,
        size,
        stride,
        weight_u8,
        weight_scale,
        input_global_scale,
        alpha,
        dtype,
        pre_quant_scale=None,
        requires_grad=False,
        layout=_NVFP4_LAYOUT_LEGACY,
        allow_kernel=True,
    ):
        super().__init__(qtype, axis)
        self._data = weight_u8
        self._scale = weight_scale
        self._input_global_scale = input_global_scale
        self._alpha = alpha
        self._pre_quant_scale = pre_quant_scale
        self._block_size = 16
        self._layout = layout
        self._allow_kernel = allow_kernel

    def __repr__(self):
        return f"NVFP4WeightTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, layout={self._layout})"

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return _dequantize_nvfp4_weight(
            weight_u8=self._data,
            weight_scale=self._scale,
            input_global_scale=self._input_global_scale,
            alpha=self._alpha,
            dtype=dtype,
            device=device,
            block_size=self._block_size,
            layout=self._layout,
        )

    def get_quantized_subtensors(self):
        if self._layout == _NVFP4_LAYOUT_TENSORCORE:
            subtensors = [
                ("weight_u8", self._data),
                ("weight_scale", self._scale),
                ("weight_scale_2", self._alpha),
                ("input_scale", self._input_global_scale),
            ]
        else:
            subtensors = [
                ("weight_u8", self._data),
                ("weight_scale", self._scale),
                ("input_global_scale", self._input_global_scale),
                ("alpha", self._alpha),
            ]
        if self._pre_quant_scale is not None:
            subtensors.append(("pre_quant_scale", self._pre_quant_scale))
        return subtensors

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        data = sub_map.get("weight_u8", sub_map.get("data"))
        if data is not None:
            self._data = data
        if "weight_scale" in sub_map and sub_map["weight_scale"] is not None:
            self._scale = sub_map["weight_scale"]
        if "input_scale" in sub_map and sub_map["input_scale"] is not None:
            self._input_global_scale = sub_map["input_scale"]
        elif "input_global_scale" in sub_map and sub_map["input_global_scale"] is not None:
            self._input_global_scale = sub_map["input_global_scale"]
        if "weight_scale_2" in sub_map and sub_map["weight_scale_2"] is not None:
            self._alpha = sub_map["weight_scale_2"]
        elif "alpha" in sub_map and sub_map["alpha"] is not None:
            self._alpha = sub_map["alpha"]
        if "pre_quant_scale" in sub_map:
            self._pre_quant_scale = sub_map["pre_quant_scale"]

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_input_global_scale", "_alpha"]
        if self._pre_quant_scale is not None:
            inner_tensors.append("_pre_quant_scale")
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "layout": self._layout,
            "allow_kernel": "1" if self._allow_kernel else "0",
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_str = meta.get("dtype", "torch.float16")
        if dtype_str.startswith("torch."):
            dtype_name = dtype_str.split(".", 1)[1]
            dtype = getattr(torch, dtype_name, torch.float16)
        else:
            dtype = getattr(torch, dtype_str, torch.float16)
        layout = meta.get("layout", _NVFP4_LAYOUT_LEGACY)
        allow_kernel = str(meta.get("allow_kernel", "1")).strip().lower() not in ("0", "false", "no")
        return NVFP4WeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            weight_u8=inner_tensors["_data"],
            weight_scale=inner_tensors["_scale"],
            input_global_scale=inner_tensors["_input_global_scale"],
            alpha=inner_tensors["_alpha"],
            pre_quant_scale=inner_tensors.get("_pre_quant_scale"),
            allow_kernel=allow_kernel,
            dtype=dtype,
            layout=layout,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, NVFP4WeightTensor):
                if _NVFP4_HAS_CUSTOM_OP:
                    # Route every NVFP4 linear through the plain-tensor custom
                    # op (torch.ops.wgp_nvfp4.gemm): at trace time the fake
                    # impl provides shapes only; at runtime the real impl runs
                    # the NVFP4 kernel (use_kernel=True) or the dequantized
                    # fallback. A graph break instead can let inductor bake the
                    # trace-time empty tensor into the compiled graph (black
                    # video, no computation executed at runtime).
                    return torch.ops.wgp_nvfp4.gemm(
                        input,
                        weight._data,
                        weight._scale,
                        weight._input_global_scale,
                        weight._alpha,
                        bias,
                        _NVFP4_LAYOUT_ID.get(_nvfp4_layout(weight), 0),
                        _NVFP4_BACKEND_ID.get(_NVFP4_KERNEL_BACKEND, 0),
                        _nvfp4_can_use_kernel(input, weight),
                    )
                return _nvfp4_linear(input, weight, bias=bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, NVFP4WeightTensor):
                if _NVFP4_HAS_CUSTOM_OP:
                    # Same plain-tensor custom op routing as __torch_function__:
                    # keeps a real opaque node in the compiled graph so the
                    # NVFP4 kernel (or the dequantized fallback) executes at
                    # runtime with real data.
                    return torch.ops.wgp_nvfp4.gemm(
                        input,
                        weight._data,
                        weight._scale,
                        weight._input_global_scale,
                        weight._alpha,
                        bias,
                        _NVFP4_LAYOUT_ID.get(_nvfp4_layout(weight), 0),
                        _NVFP4_BACKEND_ID.get(_NVFP4_KERNEL_BACKEND, 0),
                        _nvfp4_can_use_kernel(input, weight),
                    )
                return _nvfp4_linear(input, weight, bias=bias, op=op)
        if op is torch.ops.aten.detach:
            t = args[0]
            return NVFP4WeightTensor.create(
                weight_u8=op(t._data),
                weight_scale=op(t._scale),
                input_global_scale=op(t._input_global_scale),
                alpha=op(t._alpha),
                pre_quant_scale=op(t._pre_quant_scale) if t._pre_quant_scale is not None else None,
                allow_kernel=getattr(t, "_allow_kernel", True),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
                layout=t._layout,
            )
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_data = op(t._data, device=device, **(kwargs or {}))
            out_scale = op(t._scale, device=device, **(kwargs or {}))
            out_igs = op(t._input_global_scale, device=device, **(kwargs or {}))
            out_alpha = op(t._alpha, device=device, **(kwargs or {}))
            out_pre_quant_scale = op(t._pre_quant_scale, device=device, **(kwargs or {})) if t._pre_quant_scale is not None else None
            return NVFP4WeightTensor.create(
                weight_u8=out_data,
                weight_scale=out_scale,
                input_global_scale=out_igs,
                alpha=out_alpha,
                pre_quant_scale=out_pre_quant_scale,
                allow_kernel=getattr(t, "_allow_kernel", True),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
                layout=t._layout,
            )
        return _nvfp4_qfallback(op, *args, **(kwargs or {}))


class QLinearNVFP4(QModuleMixin, torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._nvfp4_default_dtype = dtype

    @classmethod
    def qcreate(
        cls,
        module,
        weights,
        activations=None,
        optimizer=None,
        device=None,
    ):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def set_default_dtype(self, dtype):
        self._nvfp4_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _NVFP4_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pre_quant_scale = getattr(self.qweight, "_pre_quant_scale", None)
        if pre_quant_scale is not None:
            input = input * pre_quant_scale.to(device=input.device, dtype=input.dtype)
        qweight = self.qweight
        if getattr(qweight, "_mm_cpu_compute", False) and qweight._data.device.type == "cpu":  # a block that MMGP leaves on the CPU (cpuCompute)
            from shared.kernels import cpu_linear
            return cpu_linear.linear(qweight, input, self.bias, input.dtype)
        return torch.nn.functional.linear(input, qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _NVFP4_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        scale_key = prefix + "weight_scale"
        scale2_key = prefix + "weight_scale_2"
        pre_quant_scale_key = prefix + "pre_quant_scale"
        igs_key = prefix + "input_global_scale"
        alpha_key = prefix + "alpha"
        input_absmax_key = prefix + "input_absmax"
        weight_global_scale_key = prefix + "weight_global_scale"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_u8 = state_dict.pop(weight_key, None)
        weight_scale = state_dict.pop(scale_key, None)
        weight_scale_2 = state_dict.pop(scale2_key, None)
        pre_quant_scale = state_dict.pop(pre_quant_scale_key, None)
        input_global_scale = state_dict.pop(igs_key, None)
        alpha = state_dict.pop(alpha_key, None)
        input_absmax = state_dict.pop(input_absmax_key, None)
        weight_global_scale = state_dict.pop(weight_global_scale_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_u8 is None:
            missing_keys.append(weight_key)
        if weight_scale is None:
            missing_keys.append(scale_key)
        layout = _NVFP4_LAYOUT_LEGACY
        allow_kernel = True
        if weight_scale_2 is not None or input_scale is not None:
            layout = _NVFP4_LAYOUT_TENSORCORE
            if weight_scale_2 is None:
                missing_keys.append(scale2_key)
            if input_scale is None:
                if torch.is_tensor(weight_scale_2):
                    input_scale = torch.ones((), dtype=torch.float32, device=weight_scale_2.device)
                elif torch.is_tensor(weight_u8):
                    input_scale = torch.ones((), dtype=torch.float32, device=weight_u8.device)
                else:
                    input_scale = torch.ones((), dtype=torch.float32)
        else:
            if input_global_scale is None or alpha is None:
                if input_absmax is not None and weight_global_scale is not None:
                    input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                    alpha = 1.0 / (input_global_scale * weight_global_scale.to(torch.float32))
                else:
                    if input_global_scale is None:
                        missing_keys.append(igs_key)
                    if alpha is None:
                        missing_keys.append(alpha_key)

        target_dtype = self._nvfp4_default_dtype or self.weight.dtype
        if layout == _NVFP4_LAYOUT_TENSORCORE:
            if weight_u8 is not None and weight_scale is not None and weight_scale_2 is not None and input_scale is not None:
                nvfp4_weight = NVFP4WeightTensor.create(
                    weight_u8=weight_u8,
                    weight_scale=weight_scale,
                    input_global_scale=input_scale,
                    alpha=weight_scale_2,
                    pre_quant_scale=pre_quant_scale,
                    allow_kernel=allow_kernel,
                    size=self.weight.size(),
                    stride=self.weight.stride(),
                    dtype=target_dtype,
                    device=weight_u8.device,
                    requires_grad=False,
                    layout=layout,
                )
                self.weight = torch.nn.Parameter(nvfp4_weight, requires_grad=False)
        else:
            if weight_u8 is not None and weight_scale is not None and input_global_scale is not None and alpha is not None:
                nvfp4_weight = NVFP4WeightTensor.create(
                    weight_u8=weight_u8,
                    weight_scale=weight_scale,
                    input_global_scale=input_global_scale,
                    alpha=alpha,
                    pre_quant_scale=pre_quant_scale,
                    size=self.weight.size(),
                    stride=self.weight.stride(),
                    dtype=target_dtype,
                    device=weight_u8.device,
                    requires_grad=False,
                    layout=layout,
                )
                self.weight = torch.nn.Parameter(nvfp4_weight, requires_grad=False)

        if bias is not None:
            if target_dtype is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
            self.bias = torch.nn.Parameter(bias)

        if torch.is_tensor(weight_u8):
            scale_device = weight_u8.device
        elif torch.is_tensor(self.weight):
            scale_device = self.weight.device
        elif torch.is_tensor(bias):
            scale_device = bias.device
        else:
            scale_device = torch.device("cpu")

        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        else:
            if not hasattr(self, "input_scale") or self.input_scale.is_meta:
                scale_dtype = self.input_scale.dtype if hasattr(self, "input_scale") else torch.float32
                self.input_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        else:
            if not hasattr(self, "output_scale") or self.output_scale.is_meta:
                scale_dtype = self.output_scale.dtype if hasattr(self, "output_scale") else torch.float32
                self.output_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        return


def validate_nvfp4_kernel(
    state_dict=None,
    checkpoint_path=None,
    device=None,
    max_layers=4,
    seed=0,
    batch_size=2,
    dtype=torch.bfloat16,
    verbose=True,
):
    """Compare kernel vs fallback outputs for a few NVFP4 layers."""
    if state_dict is None:
        if checkpoint_path is None:
            raise ValueError("state_dict or checkpoint_path is required")
        from mmgp import safetensors2

        state_dict = {}
        with safetensors2.safe_open(checkpoint_path, framework="pt", device="cpu", writable_tensors=False) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    specs = _collect_nvfp4_specs(state_dict)
    if not specs:
        return {"ok": False, "reason": "no nvfp4 weights found"}

    candidates = sorted(specs, key=lambda spec: spec["weight"].numel())
    if isinstance(max_layers, int) and max_layers > 0:
        candidates = candidates[:max_layers]

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results = []
    with torch.no_grad():
        for spec in candidates:
            weight = spec["weight"]
            layout = spec.get("layout", _NVFP4_LAYOUT_LEGACY)
            in_features = weight.shape[1] * 2
            bias = spec.get("bias")

            if layout == _NVFP4_LAYOUT_TENSORCORE:
                input_scale = spec.get("input_scale")
                tensor_scale = spec.get("weight_scale_2")
            else:
                input_scale = spec.get("input_global_scale")
                tensor_scale = spec.get("alpha")

            if input_scale is None or tensor_scale is None:
                results.append({"name": spec["name"], "layout": layout, "kernel": False, "reason": "missing scales"})
                continue

            nvfp4_weight = NVFP4WeightTensor.create(
                weight_u8=weight,
                weight_scale=spec["weight_scale"],
                input_global_scale=input_scale,
                alpha=tensor_scale,
                size=(weight.shape[0], in_features),
                stride=(in_features, 1),
                dtype=dtype,
                device=device,
                requires_grad=False,
                layout=layout,
            )

            x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
            pre_quant_scale = spec.get("pre_quant_scale")
            if pre_quant_scale is not None:
                x = x * pre_quant_scale.to(device=device, dtype=dtype)
            if bias is not None:
                bias = bias.to(device=device, dtype=dtype)

            kernel_ok = _nvfp4_can_use_kernel(x, nvfp4_weight)
            y_kernel = _nvfp4_linear_cuda(x, nvfp4_weight, bias=bias) if kernel_ok else None
            x_ref = x
            if (
                layout == _NVFP4_LAYOUT_TENSORCORE
                and _ck_cuda_available
                and device.type == "cuda"
                and torch.is_tensor(input_scale)
            ):
                input_scale_fp32 = input_scale.to(device=device, dtype=torch.float32)
                pad_16x = (x.shape[0] % 16) != 0
                qx, qx_scale = _ck_cuda.quantize_nvfp4(x, input_scale_fp32, 0.0, pad_16x)
                x_ref = _ck_cuda.dequantize_nvfp4(qx, input_scale_fp32, qx_scale, output_type=dtype)
                if pad_16x:
                    x_ref = x_ref[: x.shape[0]]
            y_ref = torch.nn.functional.linear(
                x_ref,
                nvfp4_weight.dequantize(dtype=dtype, device=device),
                bias,
            )

            if y_kernel is None:
                results.append({"name": spec["name"], "layout": layout, "kernel": False})
                continue

            diff = (y_kernel - y_ref).float()
            results.append(
                {
                    "name": spec["name"],
                    "layout": layout,
                    "kernel": True,
                    "max_abs": diff.abs().max().item(),
                    "mean_abs": diff.abs().mean().item(),
                }
            )

    if verbose:
        print("NVFP4 kernel validation:")
        for entry in results:
            if not entry.get("kernel"):
                print(f"  {entry['name']}: kernel skipped ({entry.get('reason', 'incompatible')})")
                continue
            print(
                f"  {entry['name']}: max_abs={entry['max_abs']:.6f} mean_abs={entry['mean_abs']:.6f}"
            )

    return {"ok": True, "results": results}
