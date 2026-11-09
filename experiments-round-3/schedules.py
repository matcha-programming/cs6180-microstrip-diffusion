"""Explicit learning rates indexed by successful optimizer updates, not epochs."""
from dataclasses import asdict, dataclass
import math

NAMES = ("constant", "warmup-constant", "warmup-cosine", "warmup-multistep", "warmup-plateau")


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    total_updates: int
    warmup_updates: int = 0
    peak_lr: float = 1e-4
    min_lr: float = 1e-6
    start_factor: float = .1
    milestones: tuple = (.6, .85)
    drop_factor: float = .1
    plateau_factor: float = .5
    plateau_patience: int = 2
    plateau_threshold: float = .001

    def __post_init__(self):
        if self.name not in NAMES or self.total_updates < 2:
            raise ValueError("Unknown schedule or insufficient update budget")
        if not 0 <= self.warmup_updates < self.total_updates - 1:
            raise ValueError("Warmup must leave at least two subsequent updates")
        if self.name == "constant" and self.warmup_updates:
            raise ValueError("The constant control has no warmup")
        if not all(math.isfinite(x) for x in (self.min_lr, self.peak_lr, self.start_factor)) or not 0 < self.min_lr <= self.peak_lr:
            raise ValueError("Require finite 0 < minimum LR <= peak LR")
        if not 0 < self.start_factor <= 1 or self.peak_lr * self.start_factor < self.min_lr:
            raise ValueError("Warmup start must lie between minimum and peak LR")
        if not 0 < self.drop_factor < 1 or not 0 < self.plateau_factor < 1 or self.plateau_patience < 0:
            raise ValueError("Invalid decay factors/patience")
        if not 0 <= self.plateau_threshold < 1:
            raise ValueError("Invalid plateau threshold")
        if tuple(sorted(set(self.milestones))) != tuple(self.milestones) or not all(0 < x < 1 for x in self.milestones):
            raise ValueError("Milestones must be unique increasing fractions in (0,1)")


class Schedule:
    """Set next_lr before an update, advance after success, observe after field validation.

    No scheduler tick occurs when GradScaler skips an update. Plateau patience counts
    bad validation observations; patience=2 reduces after the third consecutive one.
    """

    def __init__(self, spec):
        self.spec = spec
        self.completed = 0
        self.plateau_lr = spec.peak_lr
        self.best = None
        self.bad_checks = 0

    @property
    def next_lr(self):
        s = self.spec
        k = min(self.completed + 1, s.total_updates)
        if k <= s.warmup_updates:
            if s.warmup_updates == 1:
                return s.peak_lr
            fraction = (k - 1) / (s.warmup_updates - 1)
            return s.peak_lr * (s.start_factor + (1 - s.start_factor) * fraction)
        if s.name == "warmup-cosine":
            fraction = (k - s.warmup_updates - 1) / (s.total_updates - s.warmup_updates - 1)
            return s.min_lr + .5 * (s.peak_lr - s.min_lr) * (1 + math.cos(math.pi * fraction))
        if s.name == "warmup-multistep":
            drops = sum(k > int(f * s.total_updates) for f in s.milestones)
            return max(s.min_lr, s.peak_lr * s.drop_factor ** drops)
        if s.name == "warmup-plateau":
            return self.plateau_lr
        return s.peak_lr

    def apply(self, optimizer):
        for group in optimizer.param_groups:
            group["lr"] = self.next_lr
        return self.next_lr

    def advance(self):
        if self.completed >= self.spec.total_updates:
            raise ValueError("Update horizon exhausted")
        self.completed += 1

    def observe(self, value):
        if self.spec.name != "warmup-plateau" or self.completed < max(1, self.spec.warmup_updates):
            return
        if not math.isfinite(value) or value < 0:
            raise ValueError("Field validation MSE must be finite and nonnegative")
        if self.best is None or value < self.best * (1 - self.spec.plateau_threshold):
            self.best, self.bad_checks = value, 0
        else:
            self.bad_checks += 1
            if self.bad_checks > self.spec.plateau_patience:
                self.plateau_lr = max(self.spec.min_lr, self.plateau_lr * self.spec.plateau_factor)
                self.bad_checks = 0

    def state_dict(self):
        return {"spec": asdict(self.spec), "completed": self.completed, "plateau_lr": self.plateau_lr,
                "best": self.best, "bad_checks": self.bad_checks}

    def load_state_dict(self, state):
        if state["spec"] != asdict(self.spec):
            raise ValueError("Resume cannot change the schedule, warmup or update horizon")
        if not 0 <= state["completed"] <= self.spec.total_updates:
            raise ValueError("Invalid saved scheduler update count")
        self.completed, self.plateau_lr = state["completed"], state["plateau_lr"]
        self.best, self.bad_checks = state["best"], state["bad_checks"]
