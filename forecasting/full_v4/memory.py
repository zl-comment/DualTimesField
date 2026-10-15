"""Use the audited V3 calendar/availability policy, without its fixed fields."""
from forecasting.full_v3.memory import HistoricalMemory, load_availability

__all__ = ['HistoricalMemory', 'load_availability']
