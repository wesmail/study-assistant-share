"""The worker beside the server: restarted after a crash, stopped with the app, jobs put back."""
import sys
import threading
import time

from app.config import Settings
from app.db.connection import connect
from app.supervisor import LOCKED, WorkerSupervisor, release_interrupted


def settings(tmp_path) -> Settings:
    return Settings(library_dir=tmp_path / "library", data_dir=tmp_path / "data")


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def run_until(supervisor: WorkerSupervisor, done, timeout: float = 10) -> None:
    supervisor.start()
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        time.sleep(0.05)


def test_a_crashed_worker_is_started_again(tmp_path):
    lines = []
    supervisor = WorkerSupervisor(settings(tmp_path), child("print('hello'); raise SystemExit(3)"),
                                  cwd=tmp_path, delays=(0.05,), output=lines.append)
    run_until(supervisor, lambda: supervisor.starts >= 3)
    supervisor.stop()
    assert supervisor.starts >= 3
    assert "[worker] hello" in lines
    assert any("stopped unexpectedly (exit 3)" in line for line in lines)


def test_a_worker_that_exits_cleanly_is_left_stopped(tmp_path):
    lines = []
    supervisor = WorkerSupervisor(settings(tmp_path), child("pass"), cwd=tmp_path, delays=(0.05,),
                                  output=lines.append)
    run_until(supervisor, lambda: "[worker] exited" in lines)
    time.sleep(0.3)
    supervisor.stop()
    assert supervisor.starts == 1


def test_a_held_lock_waits_its_own_delay(tmp_path):
    lines = []
    supervisor = WorkerSupervisor(settings(tmp_path), child(f"raise SystemExit({LOCKED})"), cwd=tmp_path,
                                  delays=(0.01,), locked_delay=30, output=lines.append)
    run_until(supervisor, lambda: any("another worker" in line for line in lines))
    time.sleep(0.3)
    supervisor.stop()
    assert supervisor.starts == 1


def test_stop_interrupts_the_worker(tmp_path):
    lines = []
    code = ("import time\ntry:\n    print('ready', flush=True); time.sleep(60)\n"
            "except KeyboardInterrupt:\n    print('interrupted', flush=True)")
    supervisor = WorkerSupervisor(settings(tmp_path), child(code), cwd=tmp_path, output=lines.append)
    run_until(supervisor, lambda: "[worker] ready" in lines)
    started = time.monotonic()
    supervisor.stop()
    assert time.monotonic() - started < 5
    assert "[worker] interrupted" in lines and not supervisor.running()


def test_a_worker_that_ignores_the_stop_is_killed_and_its_job_put_back(tmp_path):
    config = settings(tmp_path)
    conn = connect(config.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, kind) VALUES (1, 'a.pdf', 'x', 'pdf')")
    conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order, status, attempts) "
                 "VALUES (1, 'x', 'parse', 0, 'running', 1)")
    conn.commit()
    lines = []
    code = ("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "print('ready', flush=True)\ntime.sleep(60)")
    supervisor = WorkerSupervisor(config, child(code), cwd=tmp_path, grace=0.5, output=lines.append)
    run_until(supervisor, lambda: "[worker] ready" in lines)
    supervisor.stop()
    assert not supervisor.running()
    row = conn.execute("SELECT status, attempts FROM jobs").fetchone()
    assert (row["status"], row["attempts"]) == ("pending", 0)


def test_release_touches_only_running_jobs(tmp_path):
    config = settings(tmp_path)
    conn = connect(config.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, kind) VALUES (1, 'a.pdf', 'x', 'pdf')")
    for stage, status, attempts in (("parse", "done", 1), ("formulas", "running", 2), ("chunk", "pending", 0)):
        conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order, status, attempts) VALUES (1, 'x', ?, 0, ?, ?)",
                     (stage, status, attempts))
    conn.commit()
    assert release_interrupted(config) == 1
    rows = {r["stage"]: (r["status"], r["attempts"]) for r in conn.execute("SELECT * FROM jobs")}
    assert rows == {"parse": ("done", 1), "formulas": ("pending", 1), "chunk": ("pending", 0)}
