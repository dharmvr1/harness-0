#!/usr/bin/env python3
"""
Debug the Docker sandbox by adding verbose output
"""
import sys
from pathlib import Path

# Make sure we can import our modules
sys.path.insert(0, '.')

from ce_agent.sandbox_factory import DockerSandbox

# Test Docker sandbox with verbose output
print("=== Testing Docker Sandbox with Verbose Output ===")
docker_sandbox = DockerSandbox()
print(f"Docker available: {docker_sandbox._available}")

if docker_sandbox._available:
    sandbox_dir = docker_sandbox.setup(".", "test_debug")
    print(f"Sandbox dir placeholder: {sandbox_dir}")
    print(f"Repo path stored: {docker_sandbox._repo_path}")

    # Test a simple command first
    print("\n--- Testing 'pwd' ---")
    result = docker_sandbox.run("pwd")
    print(f"pwd result: {result}")

    # Test ls command
    print("\n--- Testing 'ls' ---")
    result = docker_sandbox.run("ls -la")
    print(f"ls result status: {result.get('status')}")
    print(f"ls output length: {len(result.get('content', ''))}")
    if result.get("content"):
        lines = result['content'].strip().split('\n')
        print(f"First 5 lines: {lines[:5]}")

    # Test read command
    print("\n--- Testing 'cat Readme.md' ---")
    result = docker_sandbox.run("cat Readme.md")
    print(f"cat Readme.md result status: {result.get('status')}")
    if result.get("status") == "ok":
        content = result.get('content', '')
        print(f"Content length: {len(content)}")
        print(f"First 200 chars: {content[:200]}")
    else:
        print(f"Error: {result.get('error')}")
else:
    print("Docker not available")