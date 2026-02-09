"""Stacked adapter weights for compile-friendly inference.

Replaces the adapters library's per-forward Python dispatch with fixed-shape
buffer indexing. All adapter weights for all languages/tasks are stored in
stacked tensors, selected by a single integer index per forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SENTINEL = object()

_DEFAULT_MAX_ADAPTERS = 200  # 3 tasks * ~66 languages

# Attributes required on an Adapter instance for extraction (duck-typing).
_REQUIRED_ATTRS = (
    'adapter_down', 'adapter_up', 'original_ln_before',
    'original_ln_after', 'scaling', 'use_gating',
    'residual_before_ln', 'adapter_residual_before_ln',
)


class StackedPfeiffer(nn.Module):
    """Per-layer stacked adapter weights. No Python branching in forward."""

    def __init__(self, hidden_size, bottleneck_size, max_slots=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.register_buffer('down_weight', torch.zeros(max_slots, bottleneck_size, hidden_size))
        self.register_buffer('down_bias', torch.zeros(max_slots, bottleneck_size))
        self.register_buffer('up_weight', torch.zeros(max_slots, hidden_size, bottleneck_size))
        self.register_buffer('up_bias', torch.zeros(max_slots, hidden_size))
        self._frozen = False
        # Cached views for the active slot — set by set_active(), used by forward().
        # Avoids per-forward indexing into the stacked buffers.
        self._cur_down_w = self.down_weight[0]
        self._cur_down_b = self.down_bias[0]
        self._cur_up_w = self.up_weight[0]
        self._cur_up_b = self.up_bias[0]

    def forward(self, x):
        """x: [N, T, hidden_size] -> [N, T, hidden_size]."""
        down = F.relu(F.linear(x, self._cur_down_w, self._cur_down_b))
        return F.linear(down, self._cur_up_w, self._cur_up_b)

    def set_active(self, idx: int):
        self._cur_down_w = self.down_weight[idx]
        self._cur_down_b = self.down_bias[idx]
        self._cur_up_w = self.up_weight[idx]
        self._cur_up_b = self.up_bias[idx]

    def register_weights(self, idx, down_w, down_b, up_w, up_b):
        if self._frozen:
            raise RuntimeError(
                "Cannot register new adapter weights after freeze(). "
                "Pre-load all languages before calling freeze()/torch.compile()."
            )
        if idx >= self.down_weight.shape[0]:
            self._grow(idx + 1)
        self.down_weight[idx].copy_(down_w)
        self.down_bias[idx].copy_(down_b)
        self.up_weight[idx].copy_(up_w)
        self.up_bias[idx].copy_(up_b)

    def freeze(self):
        """Prevent further buffer growth. Call before torch.compile."""
        self._frozen = True

    def _grow(self, new_size):
        target = max(new_size, self.down_weight.shape[0] * 2)
        device, dtype = self.down_weight.device, self.down_weight.dtype
        for name in ('down_weight', 'down_bias', 'up_weight', 'up_bias'):
            old = getattr(self, name)
            new = torch.zeros(target, *old.shape[1:], device=device, dtype=dtype)
            new[:old.shape[0]] = old
            self.register_buffer(name, new)


class StackedAdapterRegistry:
    """Manages slot-to-index mapping across all layers."""

    def __init__(self, layers: list, max_adapters=_DEFAULT_MAX_ADAPTERS):
        self.layers = layers
        self.slot_to_idx: dict = {}
        self.next_idx = 0
        self.max_adapters = max_adapters

    def register(self, slot_name, adapter_weights_per_layer):
        idx = self.slot_to_idx.get(slot_name)
        if idx is None:
            if self.next_idx >= self.max_adapters:
                raise RuntimeError(
                    f"Stacked adapter registry full ({self.max_adapters} slots, "
                    f"~{self.max_adapters * 4.5:.0f}MB at fp16). "
                    f"Each language uses 2-3 slots (~9-13.5MB). "
                    f"Increase max_adapters or reduce loaded languages. "
                    f"Current slots: "
                    f"{list(self.slot_to_idx.keys())[:10]}{'...' if len(self.slot_to_idx) > 10 else ''}"
                )
            idx = self.next_idx
            self.next_idx += 1
            self.slot_to_idx[slot_name] = idx

        if len(adapter_weights_per_layer) != len(self.layers):
            raise ValueError(
                f"Expected weights for {len(self.layers)} layers, "
                f"got {len(adapter_weights_per_layer)}"
            )
        for stacked, weights in zip(self.layers, adapter_weights_per_layer):
            stacked.register_weights(idx, **weights)
        return idx

    def set_active(self, slot_name):
        idx = self.slot_to_idx[slot_name]
        for layer in self.layers:
            layer.set_active(idx)

    def has_slot(self, slot_name):
        return slot_name in self.slot_to_idx

    def freeze(self):
        for layer in self.layers:
            layer.freeze()


def _get_adapter_attr(adapter, primary, *alternates, default=_SENTINEL):
    """Get an adapter attribute, falling back to alternates."""
    for name in (primary, *alternates):
        if hasattr(adapter, name):
            return getattr(adapter, name), name
    if default is not _SENTINEL:
        return default, None
    raise AttributeError(f"Missing attribute: tried {(primary, *alternates)}")


def validate_adapter_instance(adapter):
    """Validate an Adapter instance matches stacked forward assumptions.

    Returns (ok, reason) — reason is None on success, error string on failure.
    """
    checks = [
        ('original_ln_before', [], True),
        ('original_ln_after', [], True),
        ('residual_before_ln', [], True),
        ('adapter_residual_before_ln', [], False),
        ('add_layer_norm_before', ['ln_before'], False),
        ('add_layer_norm_after', ['ln_after'], False),
        ('use_gating', [], False),
    ]
    for primary, alternates, expected in checks:
        try:
            actual, used_name = _get_adapter_attr(adapter, primary, *alternates)
        except AttributeError:
            if expected in (False, None, 0):
                continue
            return False, f"{primary} not found (tried {[primary]+alternates}), expected {expected}"
        if actual != expected:
            name_info = f" (via {used_name})" if used_name != primary else ""
            return False, f"{primary}{name_info}={actual}, expected {expected}"

    # scaling must be 1.0 (float, not learned)
    scaling = getattr(adapter, 'scaling', 1.0)
    if not isinstance(scaling, (int, float)) or float(scaling) != 1.0:
        return False, f"scaling={scaling}, expected 1.0"
    # no stochastic depth
    if hasattr(adapter, 'DropPath'):
        return False, "has DropPath (stochastic_depth > 0)"
    return True, None


def _has_adapter_attrs(obj):
    """Check if obj has all required Adapter attributes (duck-typing)."""
    return hasattr(obj, '__dict__') and all(hasattr(obj, attr) for attr in _REQUIRED_ATTRS)


def _resolve_adapter(module, slot_name, layer_idx):
    """Get and resolve the Adapter module from a BottleneckLayer.

    Uses attribute duck-typing — works regardless of class hierarchy changes.
    """
    raw = module.get_adapter(slot_name)
    if raw is None:
        raise RuntimeError(
            f"Adapter '{slot_name}' not found on layer {layer_idx} output. "
            f"Available: {list(module.adapter_modules.keys())}"
        )

    # Direct hit
    if _has_adapter_attrs(raw):
        return raw

    # Unwrap common container types
    if isinstance(raw, dict):
        for v in raw.values():
            if _has_adapter_attrs(v):
                return v
        for v in raw.values():
            if isinstance(v, nn.Module):
                for child in v.modules():
                    if _has_adapter_attrs(child):
                        return child
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if _has_adapter_attrs(item):
                return item

    # nn.Module tree walk
    if isinstance(raw, nn.Module):
        for child in raw.modules():
            if child is not raw and _has_adapter_attrs(child):
                return child

    # Hard fail with introspection dump
    present = [a for a in _REQUIRED_ATTRS if hasattr(raw, a)]
    missing = [a for a in _REQUIRED_ATTRS if not hasattr(raw, a)]
    children_info = ""
    if isinstance(raw, nn.Module):
        children_info = (f"\n  children: "
                         f"{[(type(c).__name__, _has_adapter_attrs(c)) for c in raw.children()]}")
    raise RuntimeError(
        f"Layer {layer_idx}: get_adapter('{slot_name}') returned {type(raw).__name__}, "
        f"which is missing required adapter attributes.\n"
        f"  type: {type(raw).__module__}.{type(raw).__qualname__}\n"
        f"  present: {present}\n"
        f"  missing: {missing}"
        f"{children_info}\n"
        f"Adapters library version may be incompatible."
    )


def extract_adapter_weights(xlmr_model, slot_name):
    """Extract per-layer adapter weights via direct module access.

    Returns:
        (weights_per_layer, hidden_size, bottleneck_size)
        where weights_per_layer is list of dicts with keys:
        'down_w', 'down_b', 'up_w', 'up_b'
    """
    weights_per_layer = []
    hidden_size = None
    bottleneck_size = None

    for i, layer in xlmr_model.iter_layers():
        output_module = layer.output
        adapter = _resolve_adapter(output_module, slot_name, i)

        ok, reason = validate_adapter_instance(adapter)
        if not ok:
            raise ValueError(
                f"Adapter '{slot_name}' layer {i} incompatible with stacked mode: {reason}. "
                f"Stacked adapters require default Pfeiffer (SeqBnConfig) configuration."
            )

        # adapter_down is nn.Sequential([Linear, ReLU]) for default Pfeiffer (ln_before=False)
        # The Linear is at index 0
        down_linear = adapter.adapter_down[0]
        down_w = down_linear.weight.data  # [bottleneck, hidden]
        down_b = down_linear.bias.data    # [bottleneck]
        up_w = adapter.adapter_up.weight.data   # [hidden, bottleneck]
        up_b = adapter.adapter_up.bias.data     # [hidden]

        # Infer/validate dimensions from actual weight shapes
        bn = down_w.shape[0]
        hs = down_w.shape[1]
        if hidden_size is None:
            hidden_size, bottleneck_size = hs, bn
        elif (hs, bn) != (hidden_size, bottleneck_size):
            raise RuntimeError(
                f"Adapter dimension mismatch at layer {i}: "
                f"({hs}, {bn}) vs expected ({hidden_size}, {bottleneck_size})"
            )

        weights_per_layer.append({
            'down_w': down_w, 'down_b': down_b,
            'up_w': up_w, 'up_b': up_b,
        })

    return weights_per_layer, hidden_size, bottleneck_size


def install_stacked_adapters(xlmr_model, hidden_size, bottleneck_size):
    """Replace bottleneck_layer_forward on each output layer with stacked adapter forward.

    Returns list of StackedPfeiffer modules (one per layer).
    """
    device = next(xlmr_model.parameters()).device
    dtype = next(xlmr_model.parameters()).dtype
    stacked_layers = []

    for i, layer in xlmr_model.iter_layers():
        output_module = layer.output
        stacked = StackedPfeiffer(hidden_size, bottleneck_size)
        stacked.to(device=device, dtype=dtype)
        output_module._stacked_pfeiffer = stacked
        stacked_layers.append(stacked)

        def make_replacement(mod):
            def _stacked_forward(hidden_states, residual_input, layer_norm):
                if residual_input is not None:
                    ffn_out = hidden_states
                    # pre-adapter LN (original_ln_before=True, residual_before_ln=True)
                    h = layer_norm(ffn_out + residual_input)
                    # adapter: down -> relu -> up (inside StackedPfeiffer.forward)
                    up = mod._stacked_pfeiffer(h)
                    # residual to FFN output (adapter_residual_before_ln=False)
                    adapter_out = up + ffn_out
                    # post-adapter LN (original_ln_after=True)
                    return layer_norm(adapter_out + residual_input)
                elif layer_norm is not None:
                    return layer_norm(hidden_states)
                return hidden_states
            return _stacked_forward
        output_module.bottleneck_layer_forward = make_replacement(output_module)

    return stacked_layers
