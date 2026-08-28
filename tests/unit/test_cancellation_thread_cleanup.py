"""Test to verify that worker thread terminates and cleans up upon request cancellation on Windows."""

import http.server
import socket
import socketserver
import threading
import time
from typing import Set

import pytest

from x4_advisor.llm.client import OllamaCancelledError, OllamaClient


class SlowBlockingHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that reads the request and blocks without writing a response."""

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len > 0:
            self.rfile.read(content_len)

        # Block indefinitely until server shutdown or connection closed
        # Sleep in short increments to hold the socket open
        for _ in range(100):
            time.sleep(0.1)


def run_server(server: socketserver.TCPServer):
    try:
        server.serve_forever()
    except Exception:
        pass


def test_cancellation_actually_terminates_worker_thread():
    """Empirically tests that when cancellation occurs, the background worker thread terminates."""
    # 1. Start a slow blocking test server on an ephemeral port
    server = socketserver.TCPServer(("127.0.0.1", 0), SlowBlockingHandler)
    server_port = server.server_address[1]

    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    time.sleep(0.1)

    try:
        client = OllamaClient(
            endpoint=f"http://127.0.0.1:{server_port}",
            model_name="gemma4:12b",
            timeout_router=10.0,
            timeout_synthesizer=10.0,
        )

        cancel_event = threading.Event()
        threads_before: Set[int] = set(t.ident for t in threading.enumerate() if t.ident is not None)

        def trigger_cancel():
            time.sleep(0.5)  # Let worker thread establish connection and block on getresponse()
            cancel_event.set()

        t_cancel = threading.Thread(target=trigger_cancel, daemon=True)
        t_cancel.start()

        # 2. Confirm caller receives OllamaCancelledError
        raised = False
        try:
            client.chat(messages=[{"role": "user", "content": "hello"}], cancel_event=cancel_event)
        except OllamaCancelledError:
            raised = True
        assert raised, "Caller did not receive OllamaCancelledError"

        # 3. Check if worker thread cleanly exited
        # Give grace period for thread cleanup
        worker_cleaned_up = False
        for _ in range(30):  # Check over 3 seconds
            time.sleep(0.1)
            threads_now = set(t.ident for t in threading.enumerate() if t.ident is not None)
            leaked = (threads_now - threads_before) - {t_cancel.ident}
            if not leaked:
                worker_cleaned_up = True
                break

        threads_after = set(t.ident for t in threading.enumerate() if t.ident is not None)
        leaked_threads = (threads_after - threads_before) - {t_cancel.ident}

        print(f"\n[Thread Verification] Leaked threads: {leaked_threads}")
        for t in threading.enumerate():
            if t.ident in leaked_threads:
                print(f"  Leaked thread name: {t.name}, alive: {t.is_alive()}, daemon: {t.daemon}")

        assert not leaked_threads, f"Worker thread(s) still alive after cancellation: {leaked_threads}"

    finally:
        server.shutdown()
        server.server_close()
