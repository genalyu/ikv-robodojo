# RoboDojo Qwen IKV evaluation

Opt-in Qwen3.5-4B scoring replaces formula/DINO indices with immutable native Qwen visual features, source time and Sys1 patch coordinates. Sys1 receives ordinary RGB and its checkpoint is unchanged. The protocol implementation comes from the verified IKV Sys2 branch b350082.

DM05 reads a 320-position history prefix, protects the newest 16 history patches, and leaves its dense current three-camera prefix unchanged. It refreshes the original 20-frame window and archives every real history K/V row on CPU. Selected older rows can re-enter, with existing RoPE rebasing preserved.

PI05 reads N visual prefix positions, where N equals all baseline current visual tokens (768 for the three-camera checkpoint), plus current text. Native prefix encoding is separated from action denoising; all visual K/V is archived on CPU and Qwen selects N rows across current and historical candidates. OpenWAM uses the same N-slot rule for its clean observed video prefix. Its clean branch keeps dense self-attention; action/future queries use the selected memory. These fixed N budgets cannot protect all current N tokens simultaneously while also reading history.

Whenever the archive exceeds the GPU budget, every candidate receives a validated integer score. If all rows fit, the selector retains all rows directly: ranking cannot change attention, so unused score generation is omitted without substituting a formula. Requests are split into groups of 128 candidate IDs while each group sees all historical native frame evidence; this prevents output truncation without dropping candidates. Scores use the common absolute 0-100 scale. No formula fallback is allowed. CPU archive updates commit only after action generation and complete layer validation; scoring errors roll back. Only selected historical rows missing from the prior GPU read set transfer to GPU. Default archive hard limit is 8 GiB per stream. Motion filtering is disabled.

Qwen decoder state is not persisted. Native vision packets remain immutable in the separate Sys2 feature store. Current task instructions are sent with each request. Episode reset clears the per-stream Sys1 archive; the service's persistent feature store has a separate hard capacity.

All three are inference ablations on official RoboDojo checkpoints. Real score JSON and complete task rollouts must be validated before claiming task performance. Synchronous Qwen scoring can be very slow, especially for large PI05/OpenWAM token grids; runtime logs record candidate counts and latency. Unit tests do not establish task accuracy.
