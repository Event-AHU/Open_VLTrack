#!/usr/bin/env python3
import argparse
import json
import os
import sys


def has_error(obj):
    if isinstance(obj, dict):
        if "error" in obj:
            return True
        for v in obj.values():
            if has_error(v):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if has_error(v):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Find and optionally delete JSON files containing LLM error entries.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing per-seq JSON files.")
    parser.add_argument("--delete", action="store_true", help="Delete files that contain errors.")
    parser.add_argument("--print_files", action="store_true", help="Print bad file paths.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print("Not a directory: %s" % args.input_dir, file=sys.stderr)
        sys.exit(1)

    total_files = 0
    bad_files = 0
    bad_paths = []
    bad_load = 0

    for name in sorted(os.listdir(args.input_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(args.input_dir, name)
        total_files += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            bad_load += 1
            bad_files += 1
            bad_paths.append(path)
            continue

        if has_error(payload):
            bad_files += 1
            bad_paths.append(path)

    print("scanned_files=%d" % total_files)
    print("bad_files=%d" % bad_files)
    print("bad_load=%d" % bad_load)

    if args.print_files:
        for p in bad_paths:
            print(p)

    if args.delete:
        for p in bad_paths:
            try:
                os.remove(p)
            except Exception as e:
                print("failed_to_delete=%s error=%s" % (p, e), file=sys.stderr)


if __name__ == "__main__":
    main()
