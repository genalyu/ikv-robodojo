# RGB Max teacher and periodic DM05 history selection

`RGBMaxClient(task, root, pooling="mean")` registers each observed three-camera RGB atlas in frame order and sends the complete registered prefix to the Qwen3.8-Max compatible API. The root supplies `private/bailian.key` and `prompt_visual.txt`; credentials are never written to requests or results. Every API query retains its frame mapping, raw responses, usage, exact block assignment and decoded scores. API/validation errors raise rather than falling back to fabricated scores.

The teacher codec assigns integers to every frame/camera/8x8 patch through a default and ordered half-open cuboids. For DM05 history, register the same square-padded RGB used by native preprocessing. Head-camera 2x2 groups map to the native 4x4 grid by configurable mean or maximum. Wrist views supply context; they do not create extra history KV. Current three-view native dense KV remain separate.

`PeriodicRGBBank(..., interval=4)` archives every real native KV update. At an interval boundary it scores the complete historical candidate set and recalls the highest-ranked actual KV. Between boundaries, only resident or genuinely new IDs are eligible; refreshes cannot silently resurrect evicted patches, and capacity deficits are evicted FIFO. Cadence advances only on a successful transaction commit. Source IDs remain temporally sorted. No VLM reread or task-specific actions are introduced by this selector.

The caller owns episode reset and must construct a fresh bank/client per episode. Registered frames must be contiguous and causal. Current implementation is intended for prefixes below the provider image-count limit; it never silently drops history. Larger histories require a lossless full-resolution packet transport before use.
