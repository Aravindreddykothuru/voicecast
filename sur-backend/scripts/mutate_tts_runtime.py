"""Mutation tests for the TTS runtime: break each check on purpose, confirm
the test that guards it fails (and name it), restore it byte-for-byte.

    python scripts/mutate_tts_runtime.py            # every mutant
    python scripts/mutate_tts_runtime.py breaker    # only matching ones

Every check in app/tts_runtime/ has an entry. A MISSED line means the check
is not really tested -- fix the test, not this list. The file under test is
always restored (finally), and the restore is verified.
"""
import os
import pathlib
import subprocess
import sys
import time

B = str(pathlib.Path(__file__).resolve().parents[1])
R = "app/tts_runtime/"
NL, CRLF = chr(10), chr(13) + chr(10)
U = "tests/test_tts_runtime_units.py::"
F = "tests/test_tts_runtime_failover.py::"
C = "tests/test_tts_runtime_chaos.py::"
X = "tests/test_tts_runtime_real.py::"

MUTANTS = [
    ("allowlist: forbidden list ignored", R + "licenses.py",
     "    for pattern, why in FORBIDDEN:\n        if re.search(pattern, repo):",
     "    for pattern, why in ():\n        if re.search(pattern, repo):",
     [U + "test_forbidden_models_are_refused_with_the_reason"]),
    ("allowlist: test-only models allowed in chains", R + "licenses.py",
     "    if entry.test_only and not allow_test_only:", "    if False:",
     [U + "test_test_only_models_cannot_join_a_chain"]),
    ("allowlist: licence mismatch ignored", R + "licenses.py",
     "    if license_id != entry.license:", "    if False:",
     [U + "test_a_licence_that_differs_from_the_record_is_refused"]),
    ("config: unknown keys accepted", R + "config.py",
     "    unknown = set(d) - allowed\n    if unknown:", "    unknown = set(d) - allowed\n    if False:",
     [U + "test_config_errors_are_precise"]),
    ("config: auto_remove allowed", R + "config.py",
     '    if k.get("auto_remove", False):', "    if False:",
     [U + "test_config_errors_are_precise"]),
    ("store: WAL not requested", R + "store.py",
     'mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0]',
     'mode = "wal"', [U + "test_store_runs_in_wal_mode_with_full_sync"]),
    ("store: synchronous not FULL", R + "store.py",
     'self._db.execute("PRAGMA synchronous=FULL")', 'self._db.execute("PRAGMA synchronous=NORMAL")',
     [U + "test_store_runs_in_wal_mode_with_full_sync"]),
    ("pins: git id not compared", R + "manifest.py",
     "    if got != pin.git_oid:", "    if False:",
     [U + "test_small_file_verifies_by_git_blob_id", U + "test_lfs_file_verifies_through_its_pointer"]),
    ("pins: LFS pointer built wrong", R + "manifest.py",
     'return f"version https://git-lfs.github.com/spec/v1\\noid sha256:{sha256_hex}\\nsize {size}\\n".encode()',
     'return f"version https://git-lfs.github.com/spec/v1\\noid sha256:{sha256_hex}\\nsize {size}".encode()',
     [U + "test_pointer_method_reproduces_published_upstream_ids"]),
    ("download: no Range on resume", R + "downloader.py",
     '            headers["Range"] = f"bytes={pos}-"', "            pass",
     [C + "test_network_cut_mid_download_resumes_from_the_partial_file"]),
    ("download: bad file kept", R + "downloader.py",
     "    except HashMismatch:\n        with contextlib.suppress(OSError):\n            part.unlink()\n        raise",
     "    except HashMismatch:\n        raise",
     [C + "test_a_download_that_fails_its_hash_is_deleted_and_never_used"]),
    ("netqueue: orphaned RUNNING tasks not recovered", R + "netqueue.py",
     "        if not self._recovered:\n            self.recover()", "        pass",
     [C + "test_a_download_killed_while_running_is_requeued_and_resumed"]),
    ("netqueue: tasks run while offline", R + "netqueue.py",
     "        if not self.netmon.is_up():\n            for t in self.store.tasks((\"QUEUED\", \"RUNNING\")):",
     "        if False:\n            for t in self.store.tasks((\"QUEUED\", \"RUNNING\")):",
     [C + "test_task_queue_pauses_offline_and_resumes_on_reconnect_with_backoff"]),
    ("trigger: duration off", R + "checks.py",
     "        elif not rng[0] <= cps <= rng[1]:", "        elif False:",
     [U + "test_each_trigger_fires_with_a_precise_detail"]),
    ("trigger: lead silence off", R + "checks.py",
     "        if lead_s > calib.max_lead_s:", "        if False:",
     [U + "test_each_trigger_fires_with_a_precise_detail"]),
    ("trigger: clipping off", R + "checks.py",
     "        if clip_frac > calib.clip_max_fraction:", "        if False:",
     [U + "test_each_trigger_fires_with_a_precise_detail"]),
    ("trigger: NaN accepted", R + "checks.py",
     "    elif not np.isfinite(x).all():", "    elif False:",
     [U + "test_each_trigger_fires_with_a_precise_detail"]),
    ("trigger: reader verdict ignored", R + "checks.py",
     "        if not any(verdicts.values()):", "        if False:",
     [U + "test_reader_trigger_and_disagreement"]),
    ("breaker: never opens on ratio", R + "breaker.py",
     "        if fails / len(recent) > self.cfg.fail_ratio:", "        if False:",
     [U + "test_breaker_opens_above_the_ratio_and_not_before_min_samples",
      F + "test_circuit_breaker_opens_skips_the_model_and_alerts"]),
    ("breaker: never half-opens", R + "breaker.py",
     '        if row["state"] == "open" and self.clock() - (row["opened_at"] or 0) >= self.cfg.cooldown_s:',
     "        if False:", [U + "test_breaker_half_opens_after_cooldown_and_closes_on_a_passing_trial"]),
    ("known-bad: learns from sentences", R + "knownbad.py",
     "        if len(ws) != 1:\n            return None", "        pass",
     [U + "test_only_syspin_single_word_lines_teach_known_bad"]),
    ("known-bad: never added", R + "knownbad.py",
     "        if fails >= self.hits:", "        if False:",
     [U + "test_word_added_after_3_of_last_5_failures_and_never_removed"]),
    ("failover: known-bad skip removed", R + "failover.py",
     "            bad = ctx.knownbad.hits_in(lang, text)\n            if bad:",
     "            bad = ctx.knownbad.hits_in(lang, text)\n            if False:",
     [F + "test_known_bad_word_goes_to_the_next_model_when_one_exists"]),
    ("failover: no retries", R + "failover.py",
     "    for try_no in range(1 + max(0, ctx.cfg.retries_per_model)):", "    for try_no in range(1):",
     [F + "test_model_1_failing_hands_the_line_to_model_2"]),
    ("failover: best attempt is the worst", R + "failover.py",
     "    best = min(with_audio, key=lambda t: t.severity()) if with_audio else None",
     "    best = max(with_audio, key=lambda t: t.severity()) if with_audio else None",
     [F + "test_every_model_failing_flags_the_line_keeps_the_best_audio_and_finishes_the_batch"]),
    ("pool: timeout reported as a plain error", R + "pool.py",
     "            raise WorkerTimeout(self.model_id, f\"{msg.get('op')} exceeded {timeout:.0f}s; worker killed\") from None",
     "            raise WorkerError(self.model_id, f\"{msg.get('op')} exceeded {timeout:.0f}s; worker killed\") from None",
     [F + "test_a_hung_model_is_killed_at_the_timeout_and_the_next_model_speaks"]),
    ("pool: OOM does not unload LRU", R + "pool.py",
     '            self.evict_lru(except_model=model_id, reason=f"OOM in {model_id}")', "            pass",
     [F + "test_oom_unloads_the_least_recently_used_model_and_retries_once"]),
    ("pool: no CPU fallback", R + "pool.py",
     "                if cls_ok and model_id not in self.cpu_forced:", "                if False:",
     [F + "test_second_oom_falls_back_to_cpu"]),
    ("health: failure does not open breaker", R + "failover.py",
     '            self.breaker.health_failed(model, lang, str(h.get("detail", "failed")))', "            pass",
     [F + "test_a_failed_health_check_opens_the_breaker_before_any_line"]),
    ("scenes: never unified", R + "runner.py",
     "            if not stopped and self.cfg.scene_consistency:", "            if False:",
     [F + "test_a_mixed_scene_is_re_rendered_with_the_highest_ranked_model_that_passes_all"]),
    ("recovery: DONE outputs not verified", R + "runner.py",
     '            if p is None or not Path(p).exists() or sha256_file(p) != ln["output_sha256"]:',
     "            if False:", [C + "test_resume_re_renders_a_done_line_whose_output_is_missing_or_changed"]),
    ("recovery: in-flight lines not requeued", R + "runner.py",
     '        for ln in self.store.lines(job_id, ("RENDERING", "CHECKING")):',
     '        for ln in self.store.lines(job_id, ("NOPE",)):',
     [C + "test_kill_9_mid_render_then_resume_finishes_with_no_duplicates_or_corrupt_files"]),
    ("disk: guard never pauses", R + "runner.py",
     "            if free >= self.cfg.min_free_disk_mb:", "            if True:",
     [C + "test_low_disk_pauses_the_batch_with_an_alert_then_resumes_without_touching_files"]),
    ("fsutil: temp file left on ENOSPC", R + "fsutil.py",
     "    except OSError as e:\n        with contextlib.suppress(OSError):\n            os.remove(tmp)",
     "    except OSError as e:\n        with contextlib.suppress(OSError):\n            pass",
     [C + "test_atomic_write_on_enospc_leaves_the_old_file_and_no_temp"]),
    ("corrupt: no re-fetch queued", R + "failover.py",
     '                if self.on_corrupt is not None:\n                    self.on_corrupt(model, h["path"])',
     "                pass", [C + "test_a_corrupted_model_file_is_caught_never_loaded_refetched_and_restored"]),
    ("offline guard: network allowed", R + "worker.py",
     "    if os.environ.get(DENY_NETWORK_ENV) == \"1\":\n        _deny_network()",
     "    if False:\n        _deny_network()", [X + "test_the_offline_guard_really_blocks_the_network"]),
    ("known-bad: learns from crashes", R + "failover.py",
     "            # it counts against the model's breaker, never toward known-bad.\n            continue",
     "            ctx.knownbad.record(model, lang, text, False)\n            continue",
     [F + "test_crashes_count_against_the_breaker_but_never_teach_known_bad"]),
    ("pool: no OOM ladder while loading", R + "pool.py",
     "        h = self._get_with_oom_ladder(model_id)\n        r = self._req(h, model_id, req)",
     "        h = self.get(model_id)\n        r = self._req(h, model_id, req)",
     [F + "test_the_pool_itself_recovers_from_an_out_of_memory_at_load"]),
    ("health: bypasses the OOM ladder", R + "pool.py",
     '            r = self._get_with_oom_ladder(model_id).request({"op": "health"}, self.cfg.synth_timeout_s)',
     '            r = self.get(model_id).request({"op": "health"}, self.cfg.synth_timeout_s)',
     [F + "test_out_of_memory_while_loading_follows_the_ladder_to_cpu"]),
    # Removing only the main loop's stop check is an equivalent mutant (the
    # disk guard checks the same flag before every line); the stop request
    # itself is what must work.
    ("signals: stop request ignored", R + "runner.py",
     "            self.store.event(\"stop_requested\", self.current.get(\"job\"), detail=\"signal\")\n        self._stop.set()",
     "            self.store.event(\"stop_requested\", self.current.get(\"job\"), detail=\"signal\")\n        pass",
     [C + "test_a_stop_signal_finishes_the_current_line_persists_and_exits_cleanly"]),
]

PY = str(next((p for p in (pathlib.Path(B) / ".venv/Scripts/python.exe",
                           pathlib.Path(B) / ".venv/bin/python") if p.exists()), sys.executable))

env = dict(os.environ, TEST_DATABASE_URL="postgresql+psycopg2://postgres:test@localhost:5434/test",
           STORAGE_BACKEND="local", PYTHONUTF8="1")
only = sys.argv[1:]
caught = total = 0
t_all = time.time()
for name, rel, old, new, tests in MUTANTS:
    if only and not any(o in name for o in only):
        continue
    total += 1
    path = os.path.join(B, rel)
    orig = open(path, "rb").read()
    src = orig.decode("utf-8")
    if CRLF in src:
        old, new = old.replace(NL, CRLF), new.replace(NL, CRLF)
    n = src.count(old)
    if n != 1:
        print(f"ANCHOR  {name}: found {n} times -- fix the mutant", flush=True)
        continue
    t0 = time.time()
    try:
        open(path, "wb").write(src.replace(old, new, 1).encode("utf-8"))
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "-x",
                            "-p", "no:cacheprovider", "--no-header", "-rf"], cwd=B, env=env,
                           capture_output=True, text=True, encoding="utf-8", timeout=900)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out, r = "TIMEOUT", None
    finally:
        open(path, "wb").write(orig)
    assert open(path, "rb").read() == orig, f"{rel} NOT restored"
    failed = [ln for ln in out.splitlines() if ln.startswith("FAILED")]
    if (r is None) or (r.returncode != 0 and failed):
        caught += 1
        print(f"CAUGHT  {name}  ({time.time() - t0:.0f}s)\n        {failed[0][:200] if failed else 'timed out (hung)'}",
              flush=True)
    else:
        print(f"MISSED  {name}  (exit {r.returncode})\n{out[-700:]}", flush=True)
print(f"\n{caught}/{total} mutants caught in {time.time() - t_all:.0f}s; all files restored")
sys.exit(0 if caught == total else 1)
