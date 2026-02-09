"""
Tests for stacked adapter mode.

Run with:
    .venv/bin/python -m pytest trankit/tests/test_stacked_adapters.py -v
"""

import os
import torch
import pytest
import trankit

_GPU_AVAILABLE = torch.cuda.is_available() or (
    hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
)

SHORT_TEXT = (
    "John Donovan from Apple Inc. announced a new product today in San Francisco. "
    "The device will be available next month."
)


def _make_pipeline(**kwargs):
    """Create Pipeline with device auto-detection."""
    return trankit.Pipeline("english", gpu=_GPU_AVAILABLE, **kwargs)


def _assert_stacked_active(p):
    """Assert the pipeline is actually in stacked mode, not silently cached."""
    assert p._stacked_adapters is True, \
        "Pipeline._stacked_adapters is False — stacked mode not active"


def test_stacked_forward_replacement_survives_delete():
    """Verify our bottleneck_layer_forward replacement persists after delete_adapter."""
    p = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p)
    # Trigger first adapter load (installs stacked adapters + deletes adapter slot)
    with torch.no_grad():
        p.tokenize(SHORT_TEXT)

    xlmr = p._embedding_layers.xlmr
    # The stacked slot name we created and then deleted
    from trankit.pipeline import _adapter_slot_name
    deleted_slot = _adapter_slot_name('tokenizer', 'english')

    for i, layer in xlmr.iter_layers():
        output = layer.output
        # Instance attribute must exist (not falling through to class method)
        assert 'bottleneck_layer_forward' in output.__dict__, \
            f"Layer {i}: instance replacement missing after delete_adapter"
        # The stacked slot we created should have been deleted
        assert deleted_slot not in output.adapter_modules, \
            f"Layer {i}: stacked slot '{deleted_slot}' should have been deleted"
        # Stacked module must exist
        assert hasattr(output, '_stacked_pfeiffer'), \
            f"Layer {i}: _stacked_pfeiffer not found"


def test_stacked_pipeline_outputs_match():
    """Full pipeline outputs must match between cached and stacked modes."""
    p_cached = _make_pipeline(cache_adapters=True)
    p_stacked = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p_stacked)

    with torch.no_grad():
        for fn_name in ['tokenize', 'posdep', 'ner', 'lemmatize']:
            r_cached = getattr(p_cached, fn_name)(SHORT_TEXT)
            r_stacked = getattr(p_stacked, fn_name)(SHORT_TEXT)
            assert r_cached == r_stacked, f"Mismatch in {fn_name}"

        # Full pipeline call
        r_cached = p_cached(SHORT_TEXT)
        r_stacked = p_stacked(SHORT_TEXT)
        assert r_cached == r_stacked, "Mismatch in full pipeline"


def test_stacked_registry_slot_management():
    """Verify slot registration and warm-path switching."""
    p = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p)
    with torch.no_grad():
        p.tokenize(SHORT_TEXT)

    registry = p._stacked_registry
    assert registry is not None, "Registry should be initialized after first load"
    assert registry.next_idx >= 1, "At least one slot should be registered"

    # Run tagger to register a second slot
    with torch.no_grad():
        p.posdep(SHORT_TEXT)

    assert registry.next_idx >= 2, "Tagger should register a second slot"

    # Warm path: calling tokenize again should not increase slot count
    old_count = registry.next_idx
    with torch.no_grad():
        p.tokenize(SHORT_TEXT)
    assert registry.next_idx == old_count, "Warm path should not register new slots"


def test_stacked_evict_is_noop():
    """evict_language_adapters should be a no-op in stacked mode."""
    p = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p)
    with torch.no_grad():
        p.tokenize(SHORT_TEXT)

    old_count = p._stacked_registry.next_idx
    p.evict_language_adapters("english")
    assert p._stacked_registry.next_idx == old_count, "Evict should be no-op for stacked"


def test_stacked_env_kill_switch(monkeypatch):
    """TRANKIT_STACKED_ADAPTERS=0 should disable stacked mode, falling back to cached."""
    import trankit.pipeline as pipeline_mod
    # Patch the module-level flag directly (avoids reload which breaks class identity)
    monkeypatch.setattr(pipeline_mod, '_STACKED_ADAPTERS', False)
    p = _make_pipeline(stacked_adapters=True)
    assert p._stacked_adapters is False, \
        "stacked_adapters should be False when env kill switch is active"
    assert p._cache_adapters is True, \
        "cache_adapters should be True as fallback"


def test_compile_model_smoke():
    """compile_model() should freeze registry and allow inference across tasks."""
    p = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p)

    # Pre-load all adapter types via inference
    with torch.no_grad():
        p(SHORT_TEXT)  # loads tokenizer, tagger, ner adapters

    # Freeze and compile
    p.compile_model()

    # Verify frozen
    assert p._stacked_registry.layers[0]._frozen is True, \
        "Layers should be frozen after compile_model()"

    # Inference should still work after compile
    with torch.no_grad():
        result = p(SHORT_TEXT)
    assert 'sentences' in result, "Pipeline should produce output after compile_model()"
    assert len(result['sentences']) >= 1, "Should have at least one sentence"

    # Verify registering new weights is blocked
    with pytest.raises(RuntimeError, match="freeze"):
        from trankit.stacked_adapter import StackedPfeiffer
        p._stacked_registry.layers[0].register_weights(
            999,
            torch.zeros(128, 768), torch.zeros(128),
            torch.zeros(768, 128), torch.zeros(768),
        )


def test_compile_model_errors():
    """compile_model() should raise clear errors for misuse."""
    # Not stacked
    p_cached = _make_pipeline(cache_adapters=True)
    with pytest.raises(RuntimeError, match="requires stacked_adapters"):
        p_cached.compile_model()

    # Stacked but no adapters loaded yet
    p_stacked = _make_pipeline(stacked_adapters=True)
    _assert_stacked_active(p_stacked)
    with pytest.raises(RuntimeError, match="before any adapters were loaded"):
        p_stacked.compile_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
