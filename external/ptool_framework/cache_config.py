"""
Cache configuration for ptool_framework.

Controls whether LLM calls are cached and how the cache behaves.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheConfig:
    """Configuration for LLM response caching."""
    
    enabled: bool = True
    """Whether caching is enabled globally."""
    
    backend: str = "pickle"
    """Cache backend: 'pickle' (persistent) or 'memory' (session only)."""
    
    cache_dir: Optional[str] = None
    """Directory for pickle cache files. None = use cachier default (~/.cachier/)."""
    
    stale_after: Optional[int] = None
    """Cache entries older than this many seconds are considered stale. None = never stale."""
    
    @classmethod
    def from_env(cls) -> "CacheConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
            PTOOL_CACHE_ENABLED: "true" or "false" (default: true)
            PTOOL_CACHE_BACKEND: "pickle" or "memory" (default: pickle)
            PTOOL_CACHE_DIR: Path to cache directory (default: None)
            PTOOL_CACHE_STALE_AFTER: Seconds until stale (default: None)
        """
        
        enabled = os.getenv("PTOOL_CACHE_ENABLED", "true").lower() == "true"
        backend = os.getenv("PTOOL_CACHE_BACKEND", "pickle")
        cache_dir = os.getenv("PTOOL_CACHE_DIR")
        stale_after_str = os.getenv("PTOOL_CACHE_STALE_AFTER")
        stale_after = int(stale_after_str) if stale_after_str else None
        
        return cls(
            enabled=enabled,
            backend=backend,
            cache_dir=cache_dir,
            stale_after=stale_after,
        )
# Global config instance
_CACHE_CONFIG: Optional[CacheConfig] = None
        
def get_cache_config() -> CacheConfig:
    """Get the global cache configuration."""
    global _CACHE_CONFIG
    if _CACHE_CONFIG is None:
        _CACHE_CONFIG = CacheConfig.from_env()
    return _CACHE_CONFIG

def set_cache_config(config: CacheConfig) -> None:
    """Set the global cache configuration."""
    global _CACHE_CONFIG
    _CACHE_CONFIG = config
    
def reset_cache_config() -> None:
    """Reset the global cache configuration to the default."""
    global _CACHE_CONFIG
    _CACHE_CONFIG = None
    
# def reload_cache_config() -> None:
#     """Reload the global cache configuration from the environment."""
#     global _CACHE_CONFIG
#     _CACHE_CONFIG = CacheConfig.from_env()
    
# def get_cache_dir() -> str: