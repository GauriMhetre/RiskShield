import { useEffect } from 'react';

export function usePolling(callback, intervalMs) {
  useEffect(() => {
    // Call the callback immediately on mount
    callback();

    // Set up the interval to call the callback every intervalMs
    const intervalId = setInterval(callback, intervalMs);

    // Cleanup function: this is crucial.
    // If this cleanup function were omitted, the setInterval would keep running
    // even after the component using this hook unmounts (e.g., user navigates away).
    // This creates a memory leak and causes React errors when the interval tries
    // to update state on a component that no longer exists in the DOM.
    return () => clearInterval(intervalId);
  }, [callback, intervalMs]);
}
