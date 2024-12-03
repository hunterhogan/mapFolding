import sys

from .clearOEIScache import clearOEIScache

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-cache":
        clearOEIScache()
    else:
        print("Usage: python -m mapFolding --clear-cache")