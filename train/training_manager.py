"""Training queue manager with GPU monitoring and resource management."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

import json
import logging
import queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil
import torch

logger = logging.getLogger("TrainingManager")

DB_PATH = Path("output_central/training_jobs.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class JobStatus(Enum):
    """Training job status."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


@dataclass
class TrainingJob:
    """Training job configuration."""
    job_id: str
    model_name: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    priority: int = 0  # Higher = more important
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class TrainingManager:
    """Manage training jobs with queue, GPU monitoring, and resource allocation."""

    def __init__(self, max_concurrent_jobs: int = 2):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = queue.PriorityQueue()
        self.running_jobs = {}
        self.init_database()
        self.gpu_monitor = GPUMonitor()
        self.start_queue_processor()

    def init_database(self):
        """Initialize SQLite database for training jobs."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS training_jobs
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           job_id
                           TEXT
                           UNIQUE,
                           model_name
                           TEXT
                           NOT
                           NULL,
                           dataset_path
                           TEXT,
                           epochs
                           INTEGER,
                           batch_size
                           INTEGER,
                           learning_rate
                           REAL,
                           validation_split
                           REAL,
                           status
                           TEXT
                           DEFAULT
                           'QUEUED',
                           current_epoch
                           INTEGER
                           DEFAULT
                           0,
                           best_loss
                           REAL,
                           best_epoch
                           INTEGER,
                           training_time
                           REAL,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           started_at
                           TIMESTAMP,
                           completed_at
                           TIMESTAMP,
                           error_message
                           TEXT,
                           metrics_json
                           TEXT
                       )
                       ''')

        # GPU monitoring table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS gpu_stats
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           gpu_id
                           INTEGER,
                           utilization
                           REAL,
                           memory_used
                           INTEGER,
                           memory_total
                           INTEGER,
                           temperature
                           REAL,
                           power_usage
                           REAL
                       )
                       ''')

        conn.commit()
        conn.close()
        logger.info(f"Training database initialized at {DB_PATH}")

    def submit_job(self, job: TrainingJob) -> bool:
        """Submit a training job to the queue."""
        try:
            # Save to database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO training_jobs
                           (job_id, model_name, dataset_path, epochs, batch_size,
                            learning_rate, validation_split, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               job.job_id, job.model_name, job.dataset_path,
                               job.epochs, job.batch_size, job.learning_rate,
                               job.validation_split, JobStatus.QUEUED.value, job.created_at
                           ))

            conn.commit()
            conn.close()

            # Add to priority queue (negative priority for min heap behavior)
            self.job_queue.put((-job.priority, job.job_id, job))
            logger.info(f"Job submitted: {job.job_id} - {job.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            return False

    def start_queue_processor(self):
        """Start background thread to process job queue."""
        processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        processor_thread.start()
        logger.info("Queue processor started")

    def _process_queue(self):
        """Process jobs from queue."""
        while True:
            try:
                # Check if we can run more jobs
                if len(self.running_jobs) < self.max_concurrent_jobs:
                    try:
                        _, job_id, job = self.job_queue.get(timeout=1)
                        self._execute_job(job)
                    except queue.Empty:
                        # Queue timeout (1s) chosen to balance responsiveness to new
                        # jobs with CPU usage from frequent polling; if no job is
                        # available, simply continue and try again on the next loop.
                        pass

                # Update GPU stats
                self.gpu_monitor.record_stats()

            except Exception as e:
                logger.error(f"Queue processor error: {e}")

    def _execute_job(self, job: TrainingJob):
        """Execute a training job."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Update status to RUNNING
            cursor.execute('''
                           UPDATE training_jobs
                           SET status     = ?,
                               started_at = ?
                           WHERE job_id = ?
                           ''', (JobStatus.RUNNING.value, datetime.now().isoformat(), job.job_id))

            conn.commit()
            conn.close()

            self.running_jobs[job.job_id] = job

            logger.info(f"Starting job: {job.job_id}")

            # This would call actual training function
            # For now, just log it
            metrics = {
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "learning_rate": job.learning_rate
            }

            self._complete_job(job.job_id, metrics)

        except Exception as e:
            self._fail_job(job.job_id, str(e))

    def _complete_job(self, job_id: str, metrics: dict[str, Any]):
        """Mark job as completed."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           UPDATE training_jobs
                           SET status       = ?,
                               completed_at = ?,
                               metrics_json = ?
                           WHERE job_id = ?
                           ''', (
                               JobStatus.COMPLETED.value,
                               datetime.now().isoformat(),
                               json.dumps(metrics),
                               job_id
                           ))

            conn.commit()
            conn.close()

            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            logger.info(f"Job completed: {job_id}")

        except Exception as e:
            logger.error(f"Failed to complete job: {e}")

    def _fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           UPDATE training_jobs
                           SET status        = ?,
                               completed_at  = ?,
                               error_message = ?
                           WHERE job_id = ?
                           ''', (
                               JobStatus.FAILED.value,
                               datetime.now().isoformat(),
                               error,
                               job_id
                           ))

            conn.commit()
            conn.close()

            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            logger.error(f"Job failed: {job_id} - {error}")

        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a job."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT job_id,
                                  model_name,
                                  status,
                                  current_epoch,
                                  best_loss,
                                  training_time,
                                  created_at,
                                  started_at,
                                  completed_at
                           FROM training_jobs
                           WHERE job_id = ?
                           ''', (job_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            return {
                "job_id": row[0],
                "model_name": row[1],
                "status": row[2],
                "current_epoch": row[3],
                "best_loss": row[4],
                "training_time": row[5],
                "created_at": row[6],
                "started_at": row[7],
                "completed_at": row[8]
            }

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def get_queue_status(self) -> dict[str, Any]:
        """Get overall queue status."""
        return {
            "queued_jobs": self.job_queue.qsize(),
            "running_jobs": len(self.running_jobs),
            "max_concurrent": self.max_concurrent_jobs,
            "gpu_status": self.gpu_monitor.get_status(),
            "can_accept_more": len(self.running_jobs) < self.max_concurrent_jobs
        }

    def get_running_jobs(self) -> list[dict[str, Any]]:
        """Get list of running jobs."""
        return [
            {
                "job_id": job.job_id,
                "model_name": job.model_name,
                "epochs": job.epochs,
                "batch_size": job.batch_size
            }
            for job in self.running_jobs.values()
        ]


class GPUMonitor:
    """Monitor GPU usage and resources."""

    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.has_gpu else 0

    def record_stats(self):
        """Record GPU statistics."""
        if not self.has_gpu:
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            for i in range(self.device_count):
                torch.cuda.set_device(i)

                mem_allocated = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)

                cursor.execute('''
                               INSERT INTO gpu_stats
                                   (gpu_id, utilization, memory_used, memory_total)
                               VALUES (?, ?, ?, ?)
                               ''', (
                                   i,
                                   (mem_allocated / mem_reserved * 100) if mem_reserved > 0 else 0,
                                   mem_allocated,
                                   mem_reserved
                               ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to record GPU stats: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current GPU status."""
        if not self.has_gpu:
            return {
                "has_gpu": False,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }

        status = {
            "has_gpu": True,
            "device_count": self.device_count,
            "devices": []
        }

        for i in range(self.device_count):
            torch.cuda.set_device(i)

            mem_allocated = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)

            status["devices"].append({
                "gpu_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_mb": mem_allocated / 1024 / 1024,
                "memory_reserved_mb": mem_reserved / 1024 / 1024,
                "utilization_percent": (mem_allocated / mem_reserved * 100) if mem_reserved > 0 else 0
            })

        status["cpu_percent"] = psutil.cpu_percent()
        status["memory_percent"] = psutil.virtual_memory().percent

        return status


# Global manager instance
manager = TrainingManager()
