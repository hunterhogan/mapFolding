def clearOEIScache() -> None:
    from .oeis import _formatFilenameCache, _pathCache
    from mapFolding import settingsOEISsequences
    """Delete all cached OEIS sequence files."""
    
    if not _pathCache.exists():
        print(f"Cache directory, {_pathCache}, not found - nothing to clear.")
        return
    else:
        for oeisID in settingsOEISsequences:
            pathFilenameCache = _pathCache / _formatFilenameCache.format(oeisID=oeisID)
            pathFilenameCache.unlink(missing_ok=True)
        
    print(f"Cache cleared from {_pathCache}.")

if __name__ == "__main__":
    clearOEIScache()