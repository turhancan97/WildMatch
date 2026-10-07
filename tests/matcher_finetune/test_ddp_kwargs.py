from wildmatch.matcher_finetune.train_by_lg_matches import ddp_kwargs


def test_trained_rdd_disables_buffer_broadcast():
    kwargs = ddp_kwargs(True).to_kwargs()
    assert kwargs["broadcast_buffers"] is False
    assert kwargs["find_unused_parameters"] is True


def test_matcher_only_keeps_former_ddp_defaults():
    kwargs = ddp_kwargs(False).to_kwargs()
    assert "broadcast_buffers" not in kwargs  # DDP default (True), as before the fix
    assert kwargs["find_unused_parameters"] is True
