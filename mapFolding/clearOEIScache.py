import shutil

from .oeis import pathCache


def clearOEIScache() -> None:
    """Delete all cached OEIS sequence files."""
    
    if not pathCache.exists():
        print(f"Cache directory, {pathCache}, not found - nothing to clear.")
        return
        
    shutil.rmtree(pathCache)
    pathCache.mkdir(parents=True, exist_ok=True)
    print(f"Cache cleared from {pathCache}.")

if __name__ == "__main__":
    clearOEIScache()