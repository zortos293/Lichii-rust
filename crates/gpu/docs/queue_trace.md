# Submission queue trace

The normal GPU fence report shows how long each fence took to reach the front. This optional
diagnostic shows what the submitting channel was doing during that interval. It changes no wait
or release decisions.

## Capture and decode

Enable `lichii_gpu::queue_trace=debug` in the log filter before startup. For a read alongside
the normal activity counters, use
`info,lichii_gpu::activity=debug,lichii_gpu::queue_trace=debug`. The default filter does not
enable the trace. Keep the log through clean shutdown if possible, or keep enough of the log
after the interval of interest for the reporter to drain it and pending releases to retire.

```sh
python3 crates/gpu/tools/queue_trace.py capture.log.gz > queue.jsonl
python3 -m unittest discover -s crates/gpu/tools
```

The decoder accepts a plain log, a gzip log, or standard input (`-`). Use one process's log
per invocation. Times are nanoseconds from the GPU capture's own clock. Join `fence` rows to
`wait` rows by capture and `wait_ids`. Capture IDs distinguish multiple GPU sessions in a process.
Add the `status` row's `unix_start_ns` to the relative timestamps to align them with the
other log lines.

## Reading the output

Every submission fence gets a row keyed by its channel, syncpoint id and value, and unique
sample id. The finish is the front reaching the increment, not the back issuing it or the
host finishing it. The six exclusive buckets sum to that interval:

- `semaphore_ns`, `syncpoint_ns`, and `fetch_ns` are suspension on the fence's own channel.
- `front_service_ns` is time in that channel's front run. It includes synchronous work inside
  that run, not only active CPU cycles.
- `runnable_ns` is time outside those states while the submission could otherwise progress.
  `runnable_other_ns` is the subset spent inside another channel's front run; don't add that
  subset to the total. The rest includes inbox and scheduling delay.
- `unattributed_ns` is the time from the last observed blocked probe to the resume when there
  is no usable readiness marker. A memory read that meets a semaphore might be a guest CPU
  write, a GPU write without usable provenance, or a value already present; it isn't counted
  as an exact GPU release.

For a semaphore wait, the trace joins the actual waiter latch to the exact renderer FIFO
position, verifies the release's address and value at the acquire's width, and reports the
matching offer and publication independently. Known queued releases can satisfy the GPU
acquire before publication. A written latch isn't ready until retirement after the write and
any mirror. Where the producer call adds an identifiable FIFO entry, the producer channel and
the time bracket of that call are included. An offer is when the back exposes an entry to
waiters, not when Vulkan recorded the entry. Missing IDs, partial mappings, unknown values
and absent publication records stay unknown. An unordered GPU latch is labeled separately.

The reporter drains up to 32,768 buffered events per tick. Producers don't wait when that
buffer fills. The `status` row reports cumulative drops and whether the trace closed cleanly.
The reporter confirms a clean close only after GPU workers have joined and no append is in
flight. A timeout or forced stop leaves a prefix, even if no drops were reported.

Any reported drop or malformed record invalidates attribution for the capture. Submissions
that never reached the front or whose front run was cut off get partial rows without guessed
buckets.
