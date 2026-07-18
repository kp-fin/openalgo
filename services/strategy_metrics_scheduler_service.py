"""
Strategy Metrics Scheduler — runs strategy_metrics_service.run_metrics_report()
once daily after both strategies' hard exits (ORB_Spread 15:15, EMA Regime
Crossover 15:00 IST), refreshing portfolio-tracking.md with the latest
Sharpe/PF/win-rate/avg-P&L/correlation numbers. Implements Karan's
"automated, not on-demand" tracking requirement (2026-07-18) -- see
agents/friday/memory/decisions.md.

Modeled on services/historify_scheduler_service.py's singleton BackgroundScheduler
pattern -- this job is a single fixed daily cron trigger with no per-user or
per-schedule state, so it's a much lighter version of that pattern.
"""

import os
import threading
from typing import Optional

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.logging import get_logger

logger = get_logger(__name__)

JOB_ID = "strategy_metrics_daily_report"


class StrategyMetricsScheduler:
    """Singleton scheduler for the daily strategy-metrics report."""

    _instance: Optional["StrategyMetricsScheduler"] = None
    _scheduler: BackgroundScheduler | None = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, db_url: str = None):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if db_url is None:
                db_url = os.getenv("DATABASE_URL", "sqlite:///db/openalgo.db")

            try:
                jobstores = {
                    "default": SQLAlchemyJobStore(
                        url=db_url, tablename="strategy_metrics_apscheduler_jobs"
                    )
                }
                self._scheduler = BackgroundScheduler(
                    jobstores=jobstores,
                    job_defaults={
                        "coalesce": True,
                        "max_instances": 1,
                        "misfire_grace_time": 3600,  # 1hr grace -- a missed run just means a stale report, not lost data
                    },
                )
                self._scheduler.start()

                self._scheduler.add_job(
                    _run_report_job,
                    trigger=CronTrigger(
                        hour=15, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"
                    ),
                    id=JOB_ID,
                    replace_existing=True,
                    name="Strategy metrics daily report",
                )

                self._initialized = True
                logger.debug("Strategy Metrics Scheduler initialized and started (daily 15:30 IST)")

            except Exception as e:
                logger.exception(f"Failed to initialize Strategy Metrics Scheduler: {e}")
                raise

    def run_now(self):
        """Manual trigger, bypassing the cron schedule -- useful for testing
        or an on-demand refresh without waiting for 15:30."""
        _run_report_job()


def _run_report_job():
    try:
        from services.strategy_metrics_service import run_metrics_report

        path = run_metrics_report()
        logger.info(f"Strategy metrics report updated: {path}")
    except Exception as e:
        logger.exception(f"Strategy metrics report run failed: {e}")


# Global scheduler instance
strategy_metrics_scheduler = StrategyMetricsScheduler()


def get_strategy_metrics_scheduler() -> StrategyMetricsScheduler:
    return strategy_metrics_scheduler


def init_strategy_metrics_scheduler(db_url: str = None):
    strategy_metrics_scheduler.init(db_url=db_url)
    return strategy_metrics_scheduler