from src.budget import BudgetTracker
from src.config_loader import load_config


def test_budget_estimates_positive():
    cfg = load_config()
    b = BudgetTracker(cfg)
    assert b.estimate_tts(1000) > 0
    assert b.estimate_stt_minutes(10) > 0


def test_budget_check_does_not_stop():
    cfg = load_config()
    b = BudgetTracker(cfg)
    b.spent_usd = b.max_usd + 1
    b.check()
