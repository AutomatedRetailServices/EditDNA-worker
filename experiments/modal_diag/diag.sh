#!/usr/bin/env bash
# Why did every temperature-0 call fail? Four 5-token Anthropic calls (well under 1 cent).
set -u
modal run temp_probe.py 2>&1 | grep '^PROBE'
