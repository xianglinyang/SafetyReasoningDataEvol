"""
Rate limiting utilities for LLM API calls.
"""

import asyncio
import time
import functools
from typing import Callable, Any, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_calls: int = 50  # Maximum concurrent calls
    calls_per_minute: Optional[int] = None  # Rate limit per minute
    calls_per_second: Optional[int] = None  # Rate limit per second
    delay_between_calls: float = 0.0  # Delay between individual calls


# def limit_async_func_call(max_size: int = 1024):
#     """
#     Decorator to limit concurrent async function calls using a semaphore.
    
#     Args:
#         max_size: Maximum number of concurrent calls allowed
#     """
#     def decorator(func: Callable) -> Callable:
#         semaphore = asyncio.Semaphore(max_size)
        
#         @functools.wraps(func)
#         async def wrapper(*args, **kwargs) -> Any:
#             async with semaphore:
#                 return await func(*args, **kwargs)
        
#         return wrapper
#     return decorator


class RateLimiter:
    """
    Rate limiter that enforces both concurrent and time-based limits.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_calls)
        self.call_times = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a call."""
        await self.semaphore.acquire()
        
        if self.config.calls_per_minute or self.config.calls_per_second:
            async with self.lock:
                now = time.time()
                
                # Remove old call times
                if self.config.calls_per_minute:
                    cutoff = now - 60
                    while self.call_times and self.call_times[0] < cutoff:
                        self.call_times.popleft()
                
                if self.config.calls_per_second:
                    cutoff = now - 1
                    while self.call_times and self.call_times[0] < cutoff:
                        self.call_times.popleft()
                
                # Check if we can make a call
                if (self.config.calls_per_minute and len(self.call_times) >= self.config.calls_per_minute) or \
                   (self.config.calls_per_second and len(self.call_times) >= self.config.calls_per_second):
                    # Wait until we can make a call
                    if self.config.calls_per_second:
                        wait_time = 1 - (now - self.call_times[0])
                    else:
                        wait_time = 60 - (now - self.call_times[0])
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = time.time()
                
                # Record this call
                self.call_times.append(now)
    
    def release(self):
        """Release the semaphore."""
        self.semaphore.release()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


def rate_limited_async_call(config: RateLimitConfig):
    """
    Decorator that applies rate limiting to async functions.
    
    Args:
        config: Rate limiting configuration
    """
    def decorator(func: Callable) -> Callable:
        rate_limiter = RateLimiter(config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with rate_limiter:
                if config.delay_between_calls > 0:
                    await asyncio.sleep(config.delay_between_calls)
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# class BatchProcessor:
#     """
#     Utility for processing large batches with rate limiting.
#     """
    
#     def __init__(self, config: RateLimitConfig):
#         self.config = config
#         self.rate_limiter = RateLimiter(config)
    
#     async def process_batch(self, items: list, processor_func: Callable, 
#                           batch_size: int = 50, delay_between_batches: float = 1.0,
#                           progress_callback: Optional[Callable] = None) -> list:
#         """
#         Process items in batches with rate limiting.
        
#         Args:
#             items: List of items to process
#             processor_func: Async function to process each item
#             batch_size: Number of items to process in each batch
#             delay_between_batches: Delay between batches
#             progress_callback: Optional callback for progress updates
            
#         Returns:
#             List of results
#         """
#         results = []
#         total_batches = (len(items) + batch_size - 1) // batch_size
        
#         if progress_callback:
#             progress_callback(0, total_batches, f"Processing {len(items)} items in {total_batches} batches")
        
#         for i in range(0, len(items), batch_size):
#             batch_items = items[i:i + batch_size]
#             batch_num = i // batch_size + 1
            
#             if progress_callback:
#                 progress_callback(batch_num, total_batches, f"Processing batch {batch_num}/{total_batches}")
            
#             # Process current batch
#             tasks = []
#             for item in batch_items:
#                 async def process_item(item=item):
#                     async with self.rate_limiter:
#                         return await processor_func(item)
#                 tasks.append(process_item())
            
#             batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
#             # Handle exceptions
#             processed_results = []
#             for j, result in enumerate(batch_results):
#                 if isinstance(result, Exception):
#                     print(f"Exception in batch {batch_num}, item {j}: {result}")
#                     processed_results.append(None)
#                 else:
#                     processed_results.append(result)
            
#             results.extend(processed_results)
            
#             # Add delay between batches (except for the last batch)
#             if i + batch_size < len(items) and delay_between_batches > 0:
#                 await asyncio.sleep(delay_between_batches)
        
#         if progress_callback:
#             progress_callback(total_batches, total_batches, "Completed processing all items")
        
#         return results


# Predefined configurations for common use cases
OPENAI_RATE_LIMIT = RateLimitConfig(
    max_calls=7000,
    calls_per_minute=7000,  # OpenAI's default rate limit
    calls_per_second=500,
    delay_between_calls=0.0
)

OPENROUTER_RATE_LIMIT = RateLimitConfig(
    max_calls=7000,
    calls_per_minute=1000,  # OpenAI's default rate limit
    calls_per_second=10,
    delay_between_calls=0.1
)

GEMINI_RATE_LIMIT = RateLimitConfig(
    max_calls=7000,
    calls_per_minute=7000,  # Gemini's default rate limit
    calls_per_second=500,
    delay_between_calls=0.0
)

CONSERVATIVE_RATE_LIMIT = RateLimitConfig(
    max_calls=20,
    calls_per_minute=1000,
    calls_per_second=3,
    delay_between_calls=0.5
) 

