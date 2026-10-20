"""Image-level routing interface shared by filename and future image routers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RouteDecision:
    route: str
    domain: str
    source: str
    confidence: Optional[float] = None


class FilenameRouter:
    def __init__(self, day_prefixes=('IMG_', 'day_'),
                 night_prefixes=('night_IMG_', 'night_'), unknown_route='error'):
        self.day_prefixes = self._prefixes(day_prefixes)
        self.night_prefixes = self._prefixes(night_prefixes)
        if unknown_route not in ('error', 'baseline'):
            raise ValueError('Unknown filename policy: ' + unknown_route)
        self.unknown_route = unknown_route

    @staticmethod
    def _prefixes(prefixes):
        if not prefixes or any(not prefix.strip() for prefix in prefixes):
            raise ValueError('Filename prefixes must be nonempty')
        return tuple(prefix.lower() for prefix in prefixes)

    def domain(self, filename):
        name = Path(filename).name.lower()
        day = name.startswith(self.day_prefixes)
        night = name.startswith(self.night_prefixes)
        if day and night:
            raise ValueError('Ambiguous day/night filename: ' + filename)
        return 'day' if day else 'night' if night else 'unknown'

    def predict(self, image, filename):
        # image is the full input tensor [1, C, H, W], before splitting into patches.
        # Filename routing ignores pixels; a future classifier can consume them here.
        domain = self.domain(filename)
        if domain == 'unknown' and self.unknown_route == 'error':
            raise ValueError('Cannot identify day/night from filename: ' + filename)
        route = 'baseline' if domain == 'unknown' else 'dora-' + domain
        return RouteDecision(route, domain, 'filename')


class FixedRouter:
    def __init__(self, route, day_prefixes, night_prefixes):
        if route not in ('baseline', 'dora-day', 'dora-night'):
            raise ValueError('Fixed router needs a resolved adapter route')
        self.route = route
        self.labels = FilenameRouter(day_prefixes, night_prefixes, 'baseline')

    def predict(self, image, filename):
        # Labels are only used for group metrics; they never change the fixed route.
        try:
            domain = self.labels.domain(filename)
        except ValueError:
            domain = 'unknown'
        return RouteDecision(self.route, domain, 'fixed')


def make_router(mode, fixed_route='baseline', day_prefixes=('IMG_', 'day_'),
                night_prefixes=('night_IMG_', 'night_'), unknown_route='error'):
    if mode == 'fixed':
        return FixedRouter(fixed_route, day_prefixes, night_prefixes)
    if mode == 'filename':
        return FilenameRouter(day_prefixes, night_prefixes, unknown_route)
    if mode == 'classifier':
        raise NotImplementedError(
            'classifier routing is reserved for a future image router; use fixed or filename')
    raise ValueError('Unknown router mode: ' + mode)
