#!/usr/bin/env python3
"""Decode lichii_gpu::queue_trace=debug logs to JSON Lines (Python 3, no dependencies)."""

import argparse
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import json
import re
import sys


BATCH = re.compile(r"GPU queue events capture=(\d+) unix_start_ns=(\d+) data=([0-9a-z_,;\-]+)")
STATUS = re.compile(r"GPU queue trace status capture=(\d+) unix_start_ns=(\d+) dropped=(\d+) end=(true|false)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WAIT = {"s": "semaphore_ns", "y": "syncpoint_ns", "f": "fetch_ns"}
ARITY = {
    "submit": 4, "front": 1, "run": 1, "stop": 1, "close": 1, "wait": 4,
    "resume": 2, "fetch_ready": 1, "sync_ready": 2, "acquire": 5, "latch": 4,
    "offer": 4, "producer": 3, "publish": 5, "retire": 1,
}


@dataclass
class Capture:
    unix_start_ns: int
    events: list = field(default_factory=list)
    dropped: int = 0
    ended: bool = False
    malformed: int = 0


def read_captures(lines):
    captures = {}
    for line in lines:
        if "GPU queue " not in line:
            continue
        line = ANSI.sub("", line)
        if match := BATCH.search(line):
            capture, epoch, data = match.groups()
            entry = captures.setdefault(capture, Capture(int(epoch)))
            for token in data.split(";"):
                try:
                    at, kind, *args = token.split(",")
                    parsed = tuple(
                        x if kind == "wait" and i == 1 and x in WAIT else int(x)
                        for i, x in enumerate(args)
                    )
                    if ARITY.get(kind) != len(parsed) or (
                        kind == "wait" and parsed[1] not in WAIT
                    ) or (kind == "acquire" and not 0 <= parsed[3] < 5):
                        raise ValueError
                    entry.events.append((int(at), kind, parsed))
                except ValueError:
                    entry.malformed += 1
        elif match := STATUS.search(line):
            capture, epoch, drops, end = match.groups()
            entry = captures.setdefault(capture, Capture(int(epoch)))
            entry.dropped = max(entry.dropped, int(drops))
            entry.ended |= end == "true"
    return captures


def predicate(op, value, payload, width):
    mask = (1 << width) - 1
    value, payload = value & mask, payload & mask
    return (
        value == payload,
        value >= payload,
        ((value - payload) & mask) < (1 << (width - 1)),
        bool(value & payload),
        not value & payload,
    )[op]


def overlaps_value(record, word, width, latch_value):
    at, size, value = record[:3]
    start, end = max(word, at), min(word + width, at + size)
    if start >= end:
        return False
    if value < 0:
        return True
    end = min(end, at + 8)
    if start >= end:
        return False
    mask = (1 << ((end - start) * 8)) - 1
    return ((value >> ((start - at) * 8)) & mask) == (
        (latch_value >> ((start - word) * 8)) & mask
    )


def semaphore(wait_id, acquire, latch, offers, writes, retired, producers, resumed):
    result = {"wait": wait_id, "source": "memory_or_unobserved", "ready_ns": None}
    if acquire is None:
        return result
    word, payload, op, wide = acquire
    result.update(device_word=None if word < 0 else word, payload=payload, op=op, wide=bool(wide))
    valid = [
        (t, fence, value, queued)
        for t, fence, value, queued in latch
        if t <= resumed and predicate(op, value, payload, 64 if wide else 32)
    ]
    if len(valid) != 1:
        return result
    at, fence, value, queued = valid[0]
    result.update(fifo=fence or None, observed_value=value)
    if not fence:
        return dict(result, source="gpu_without_fifo", ready_ns=at)

    if word < 0:
        return dict(result, source="gpu_fifo_partial_address")

    width = 8 if wide else 4
    offered = [(t, at, size, val) for t, at, size, val in offers if overlaps_value(
        (at, size, val), word, width, value
    )]
    known_offer = any(
        val >= 0 and at <= word and word + width <= at + min(size, 8)
        for _, at, size, val in offered
    )
    published = [t for t, at, size, val, ok in writes if ok and overlaps_value(
        (at, size, val), word, width, value
    )]
    result.update(
        offer_ns=min((item[0] for item in offered), default=None),
        publication_ns=retired if published and retired is not None and max(published) <= retired else None,
    )
    if producers:
        if len(producers) == 1:
            after, channel, before = producers[0]
            result.update(producer_channel=channel, enqueue_from_ns=before, enqueue_to_ns=after)
        else:
            return dict(result, source="gpu_fifo_ambiguous_producer")
    if queued and known_offer:
        return dict(result, source="gpu_fifo_offer", ready_ns=at)
    if not queued and result["publication_ns"] is not None and at <= retired <= resumed:
        return dict(result, source="gpu_fifo_publication", ready_ns=retired)
    return dict(result, source="gpu_fifo_partial_record")


class Timeline:
    def __init__(self):
        self.running = None
        self.waiting = None
        self.stop = 0
        self.intervals = []
        self.starts = []
        self.missing = False
        self.samples = []

    def add(self, start, end, kind, wait=None):
        if start < end:
            if self.intervals and start < self.intervals[-1][1]:
                self.missing = True
                return
            self.starts.append(start)
            self.intervals.append((start, end, kind, wait))

    def split(self, start, end, other_service):
        counts = dict.fromkeys((*WAIT.values(), "front_service_ns", "unattributed_ns"), 0)
        waits = []
        cursor = start
        other = 0
        index = max(0, bisect.bisect_right(self.starts, start) - 1)
        while index < len(self.intervals):
            lo, hi, kind, wait = self.intervals[index]
            if lo >= end:
                break
            overlap = min(end, hi) - max(start, lo)
            if overlap > 0:
                other += other_service(cursor, min(end, lo))
                cursor = min(end, hi)
                counts[kind] += overlap
                if wait is not None and (not waits or waits[-1] != wait):
                    waits.append(wait)
            index += 1
        counts["runnable_ns"] = end - start - sum(counts.values())
        counts["runnable_other_ns"] = other + other_service(cursor, end)
        return counts, waits


def decode(events, dropped, ended, unix_start_ns=0, malformed=0):
    submissions, fronts, acquires, fetch_ready, retires = {}, {}, {}, {}, {}
    sync_ready = defaultdict(list)
    latches, offers, writes, producers = (defaultdict(list) for _ in range(4))
    timelines = defaultdict(Timeline)
    waits = {}
    front_events = []
    complete_fronts = set()
    for at, kind, args in events:
        if kind == "submit":
            channel, syncpoint, value, sample = args
            submissions[sample] = (at, channel, syncpoint, value)
        elif kind == "front":
            fronts[args[0]] = at
            front_events.append((at, kind, args))
        elif kind == "acquire":
            acquires[args[0]] = args[1:]
        elif kind == "fetch_ready":
            fetch_ready[args[0]] = at
        elif kind == "sync_ready":
            sync_ready[args[0] << 32 | args[1]].append(at)
        elif kind == "retire":
            retires[args[0]] = at
        elif kind in ("latch", "offer", "publish", "producer"):
            table = {"latch": latches, "offer": offers, "publish": writes, "producer": producers}[kind]
            table[args[0]].append((at, *args[1:]))
        else:
            front_events.append((at, kind, args))

    if not (dropped or malformed):
        for at, kind, (channel, *args) in front_events:
            if kind == "front":
                if channel in submissions:
                    timeline = timelines[submissions[channel][1]]
                    if timeline.running is not None:
                        timeline.samples.append(channel)
                continue
            timeline = timelines[channel]
            if kind == "run":
                timeline.missing |= timeline.running is not None or timeline.waiting is not None
                timeline.running = at
            elif kind == "stop":
                if timeline.running is None:
                    timeline.missing = True
                else:
                    timeline.add(timeline.running, at, "front_service_ns")
                    complete_fronts.update(timeline.samples)
                    timeline.samples.clear()
                timeline.running, timeline.stop = None, at
            elif kind == "wait":
                timeline.missing |= timeline.waiting is not None
                timeline.waiting = (at, *args)
            elif kind == "resume":
                if timeline.waiting is None:
                    timeline.missing = True
                    continue
                requested, wait_kind, wait_id, readiness_key = timeline.waiting
                timeline.waiting = None
                start = max(requested, timeline.stop)
                key = f"{wait_kind}:{wait_id}"
                if wait_kind == "s":
                    seen = latches[wait_id]
                    fence = seen[0][1] if len(seen) == 1 else 0
                    evidence = semaphore(
                        wait_id, acquires.get(wait_id), seen, offers[fence],
                        writes[fence], retires.get(fence), producers[fence], at,
                    )
                    ready = evidence["ready_ns"]
                else:
                    if wait_kind == "f":
                        ready = fetch_ready.get(readiness_key)
                    else:
                        marks = sync_ready[readiness_key]
                        ready = marks[0] if len(marks) == 1 else None
                    evidence = {"wait": wait_id, "ready_ns": ready}
                last_unready = max(start, min(args[0], at))
                timed = ready is not None and ready <= at and (ready >= last_unready or last_unready == start)
                wait_end = max(start, ready) if timed else last_unready
                timeline.add(start, wait_end, WAIT[wait_kind], key)
                if not timed:
                    timeline.add(wait_end, at, "unattributed_ns", key)
                waits[key] = dict(
                    evidence, key=key, limiter=WAIT[wait_kind].removesuffix("_ns"),
                    channel=channel, start_ns=start, resumed_ns=at,
                    suspension_ns=wait_end - start,
                    uncertain_ns=0 if timed else at - wait_end,
                )
            elif kind == "close" and timeline.running is not None:
                timeline.missing = True

    yield {
        "type": "status", "dropped": dropped, "ended": ended, "malformed": malformed,
        "unix_start_ns": unix_start_ns or None,
        "missing_submissions": len(fronts.keys() - submissions.keys()),
    }
    for key, wait in waits.items():
        yield dict(wait, type="wait", key=key)
    service = sorted(
        (lo, hi, channel)
        for channel, timeline in timelines.items()
        for lo, hi, kind, _ in timeline.intervals
        if kind == "front_service_ns"
    )
    service_starts = [lo for lo, _, _ in service]

    def other_service(start, end, channel):
        if end <= start:
            return 0
        elapsed = 0
        index = max(0, bisect.bisect_right(service_starts, start) - 1)
        while index < len(service):
            lo, hi, other = service[index]
            if lo >= end:
                break
            if other != channel:
                elapsed += max(0, min(end, hi) - max(start, lo))
            index += 1
        return elapsed

    for sample, (start, channel, syncpoint, value) in submissions.items():
        end = fronts.get(sample)
        row = dict(type="fence", sample=sample, channel=channel, syncpoint=syncpoint,
                   value=value, submitted_ns=start, front_ns=end)
        timeline = timelines[channel]
        if dropped or malformed or timeline.missing or sample not in complete_fronts or end is None or end < start:
            yield dict(row, quality="dropped" if dropped else "partial")
        else:
            counts, wait_ids = timeline.split(start, end, lambda lo, hi: other_service(lo, hi, channel))
            yield dict(row, **counts, wait_ids=wait_ids,
                       quality="partial" if counts["unattributed_ns"] else "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="log path, .gz log, or - for stdin")
    args = parser.parse_args()
    if args.log == "-":
        captures = read_captures(sys.stdin)
    else:
        opener = gzip.open if args.log.endswith(".gz") else open
        with opener(args.log, "rt", encoding="utf-8", errors="replace") as lines:
            captures = read_captures(lines)
    if not captures:
        parser.error("no queue trace in the input")
    for capture, data in captures.items():
        for row in decode(**vars(data)):
            print(json.dumps(dict(row, capture=int(capture)), separators=(",", ":")))


if __name__ == "__main__":
    main()
