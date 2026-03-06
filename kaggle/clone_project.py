#!/usr/bin/env python3
"""Clone SAFB project script."""

import os
import subprocess
import sys

REPO_URL = "https://github.com/zyh-hnu/ANB.git"
PROJECT_DIR = "/SAFB"
BRANCH = "master"

if os.path.exists(PROJECT_DIR):
    print(f"Directory already exists: {PROJECT_DIR}")
    sys.exit(0)

subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, PROJECT_DIR], check=True)
print(f"Project cloned to: {PROJECT_DIR}")
