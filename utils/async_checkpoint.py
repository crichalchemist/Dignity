"""Asynchronous checkpoint saving with threading and queue-based management."""

from __future__ import annotations

import builtins
import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class AsyncCheckpointManager:
    """Asynchronous checkpoint manager with thread pool and queue-based processing."""

    def __init__(
            self,
            save_dir: Path,
            max_workers: int = 2,
            queue_size: int = 10,
            top_n_checkpoints: int = 3
    ):
        """Initialize asynchronous checkpoint manager.

        Parameters
        ----------
        save_dir : Path
            Directory to save checkpoints.
        max_workers : int
            Number of worker threads for checkpoint saving.
        queue_size : int
            Maximum size of the checkpoint queue.
        top_n_checkpoints : int
            Number of best checkpoints to retain.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.top_n_checkpoints = top_n_checkpoints

        # Queue for checkpoint requests
        self.checkpoint_queue = queue.Queue(maxsize=queue_size)

        # Worker threads
        self.workers = []
        self.shutdown_event = threading.Event()

        # Statistics
        self.saved_checkpoints = 0
        self.failed_checkpoints = 0
        self.queue_size_stats = []

        # Start worker threads
        self._start_workers()

        # Checkpoint tracking
        self.checkpoint_history = []
        self.lock = threading.Lock()

        logger.info(f"Started async checkpoint manager with {max_workers} workers")

    def _start_workers(self):
        """Start worker threads for checkpoint saving."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"checkpoint_worker_{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            logger.debug(f"Started checkpoint worker {i}")

    def _worker_loop(self):
        """Main loop for worker threads."""
        while not self.shutdown_event.is_set():
            try:
                # Get checkpoint request from queue with timeout
                checkpoint_data = self.checkpoint_queue.get(timeout=1.0)
                if checkpoint_data is None:  # Shutdown signal
                    break

                # Process checkpoint
                self._process_checkpoint(checkpoint_data)
                self.checkpoint_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in checkpoint worker: {e}")
                self.failed_checkpoints += 1

    def _process_checkpoint(self, checkpoint_data: dict):
        """Process a single checkpoint save request."""
        try:
            state_dict = checkpoint_data['state_dict']
            score = checkpoint_data['score']
            epoch = checkpoint_data['epoch']
            model_name = checkpoint_data['model_name']
            callback = checkpoint_data.get('callback')

            # Convert tensors to CPU to save memory
            cpu_state = {}
            for k, v in state_dict.items():
                if hasattr(v, 'cpu'):
                    cpu_state[k] = v.cpu()
                else:
                    cpu_state[k] = v

            # Generate checkpoint filename
            timestamp = int(time.time())
            checkpoint_path = self.save_dir / f"{model_name}_epoch{epoch}_score{score:.4f}_{timestamp}.pt"

            # Save checkpoint
            torch.save(cpu_state, checkpoint_path)

            # Call callback if provided
            if callback:
                callback(checkpoint_path, score, epoch)

            # Track checkpoint
            with self.lock:
                self.checkpoint_history.append({
                    'path': checkpoint_path,
                    'score': score,
                    'epoch': epoch,
                    'timestamp': timestamp
                })

                # Clean up old checkpoints to maintain top_n
                self._cleanup_old_checkpoints()

            self.saved_checkpoints += 1
            logger.info(f"Saved checkpoint: {checkpoint_path.name} (score: {score:.4f})")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            self.failed_checkpoints += 1

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints beyond top_n retention."""
        if len(self.checkpoint_history) <= self.top_n_checkpoints:
            return

        # Sort by score (descending for accuracy, ascending for loss)
        self.checkpoint_history.sort(key=lambda x: x['score'], reverse=True)

        # Remove excess checkpoints
        checkpoints_to_remove = self.checkpoint_history[self.top_n_checkpoints:]
        for checkpoint_info in checkpoints_to_remove:
            try:
                if checkpoint_info['path'].exists():
                    checkpoint_info['path'].unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_info['path'].name}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")

        # Keep only top_n in history
        self.checkpoint_history = self.checkpoint_history[:self.top_n_checkpoints]

    def save_checkpoint(
            self,
            state_dict: dict,
            score: float,
            epoch: int,
            model_name: str = "model",
            callback: Callable | None = None,
            blocking: bool = False
    ) -> bool:
        """Queue a checkpoint for asynchronous saving.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary to save.
        score : float
            Validation score for ranking.
        epoch : int
            Current training epoch.
        model_name : str
            Base name for checkpoint file.
        callback : Optional[Callable]
            Callback function to call after saving.
        blocking : bool
            Whether to wait for save to complete.

        Returns
        -------
        bool
            True if checkpoint was queued successfully.
        """
        checkpoint_data = {
            'state_dict': state_dict,
            'score': score,
            'epoch': epoch,
            'model_name': model_name,
            'callback': callback
        }

        try:
            if blocking:
                # For blocking saves, use a separate queue and wait
                temp_queue = queue.Queue()
                checkpoint_data['callback'] = lambda path, score, epoch: temp_queue.put((path, score, epoch))
                self.checkpoint_queue.put(checkpoint_data)
                # Wait for completion (with timeout)
                return temp_queue.get(timeout=30.0) is not None
            else:
                # Non-blocking: add to queue
                if not self.checkpoint_queue.full():
                    self.checkpoint_queue.put(checkpoint_data)

                    # Update queue size stats
                    with self.lock:
                        current_size = self.checkpoint_queue.qsize()
                        self.queue_size_stats.append(current_size)

                    return True
                else:
                    logger.warning("Checkpoint queue is full, skipping checkpoint")
                    return False

        except Exception as e:
            logger.error(f"Failed to queue checkpoint: {e}")
            return False

    def get_best_checkpoint(self) -> Path | None:
        """Get the path to the best checkpoint.

        Returns
        -------
        Optional[Path]
            Path to the best checkpoint, or None if no checkpoints exist.
        """
        with self.lock:
            if not self.checkpoint_history:
                return None
            # Return the highest scored checkpoint
            best_checkpoint = max(self.checkpoint_history, key=lambda x: x['score'])
            return best_checkpoint['path']

    def get_statistics(self) -> dict:
        """Get checkpoint manager statistics.

        Returns
        -------
        dict
            Dictionary containing performance statistics.
        """
        with self.lock:
            avg_queue_size = sum(self.queue_size_stats) / max(len(self.queue_size_stats), 1)

            return {
                'saved_checkpoints': self.saved_checkpoints,
                'failed_checkpoints': self.failed_checkpoints,
                'queue_utilization': avg_queue_size / max(self.queue_size, 1),
                'active_workers': len([w for w in self.workers if w.is_alive()]),
                'total_checkpoints': len(self.checkpoint_history),
                'success_rate': self.saved_checkpoints / max(
                    self.saved_checkpoints + self.failed_checkpoints, 1
                ) * 100
            }

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for all queued checkpoints to be processed.

        Parameters
        ----------
        timeout : Optional[float]
            Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if all checkpoints were processed within timeout.
        """
        try:
            self.checkpoint_queue.join()  # Wait for all tasks to complete
            return True
        except Exception:
            return False

    def shutdown(self, timeout: float = 30.0):
        """Shutdown the checkpoint manager and clean up resources.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for workers to finish.
        """
        logger.info("Shutting down async checkpoint manager...")

        # Signal workers to stop
        self.shutdown_event.set()

        # Add shutdown signals to queue
        for _ in range(self.max_workers):
            try:
                self.checkpoint_queue.put(None)
            except queue.Full:
                break

        # Wait for workers to finish
        start_time = time.time()
        for worker in self.workers:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                worker.join(timeout=remaining_time)
            else:
                break

        # Force shutdown if needed
        for worker in self.workers:
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.name}")
                worker.join(timeout=0.1)

        logger.info("Async checkpoint manager shutdown complete")

    def __del__(self):
        """Destructor to ensure proper shutdown."""
        with contextlib.suppress(builtins.BaseException):
            self.shutdown(timeout=5.0)


def create_async_checkpoint_manager(
        save_dir: Path,
        training_config
) -> AsyncCheckpointManager:
    """Create async checkpoint manager from training configuration.

    Parameters
    ----------
    save_dir : Path
        Directory to save checkpoints.
    training_config
        Training configuration with async settings.

    Returns
    -------
    AsyncCheckpointManager
        Configured async checkpoint manager.
    """
    return AsyncCheckpointManager(
        save_dir=save_dir,
        max_workers=getattr(training_config, 'checkpoint_workers', 2),
        queue_size=getattr(training_config, 'checkpoint_queue_size', 10),
        top_n_checkpoints=getattr(training_config, 'top_n_checkpoints', 3)
    )


class CheckpointCallback:
    """Callback handler for checkpoint events."""

    def __init__(self, callback_fn: Callable | None = None):
        """Initialize callback handler.

        Parameters
        ----------
        callback_fn : Optional[Callable]
            Function to call on checkpoint events.
        """
        self.callback_fn = callback_fn
        self.checkpoint_count = 0

    def __call__(self, checkpoint_path: Path, score: float, epoch: int):
        """Handle checkpoint event.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the saved checkpoint.
        score : float
            Validation score.
        epoch : int
            Training epoch.
        """
        self.checkpoint_count += 1

        if self.callback_fn:
            try:
                self.callback_fn(checkpoint_path, score, epoch)
            except Exception as e:
                logger.error(f"Checkpoint callback failed: {e}")

        logger.info(f"Checkpoint #{self.checkpoint_count} saved: {checkpoint_path.name}")
