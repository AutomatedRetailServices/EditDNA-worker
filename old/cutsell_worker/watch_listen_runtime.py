"""Request-scoped automatic Watch + Listen profile; no process env mutation."""
from contextvars import ContextVar
from functools import wraps
import os

_ACTIVE = ContextVar('cutsell_watch_listen_automatic', default=False)
MASTER = 'CUTSELL_WATCH_LISTEN_AUTOMATIC'
_PREFIX = 'CUTSELL_'
DEPENDENCIES = {
    'WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED': (),
    'WATCH_LISTEN_RELATION_DISCOVERY_ENABLED': ('WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED',),
    'WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED': (),
    'WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED': ('WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',),
    'WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED': ('WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED',),
    'BOUNDED_FINALIST_ARBITER_ENABLED': ('WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED',),
    'PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED': ('BOUNDED_FINALIST_ARBITER_ENABLED',),
    'BOUNDED_FINALIST_ARBITER_AUTHORITY_ENABLED': ('BOUNDED_FINALIST_ARBITER_ENABLED',),
    'EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED': (),
    'LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED': ('EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED',),
    'WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED': ('LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED',),
}


def _true(value):
    return str(value).strip().lower() in {'1','true','yes','on'}


def capability_enabled(key, values):
    """Preserve explicit rollback controls and standalone diagnostic behavior."""
    if not _ACTIVE.get():
        return _true(values.get(key, '0'))
    name = key.removeprefix(_PREFIX)
    if not _true(values.get(key, '1')):
        return False
    return all(capability_enabled(_PREFIX+dep, values) for dep in DEPENDENCIES.get(name, ()))


def runtime_diagnostics():
    active = _ACTIVE.get()
    rows = {}
    for name, dependencies in DEPENDENCIES.items():
        key = _PREFIX+name
        enabled = capability_enabled(key, os.environ)
        rows[name] = {
            'enabled': enabled,
            'activation': ('explicit' if key in os.environ else 'automatic' if active else 'default_off'),
            'blocked_dependencies': [dep for dep in dependencies if not capability_enabled(_PREFIX+dep, os.environ)],
        }
    return {'mode':'automatic' if active else 'explicit_flags', 'configuration_only':True, 'capabilities':rows}


def automatic_watch_listen(function):
    @wraps(function)
    def run(*args, **kwargs):
        token = _ACTIVE.set(_true(os.environ.get(MASTER, '1')))
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return run
