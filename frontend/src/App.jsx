import { useEffect } from 'react'
import { getFlags } from './api/client'

/**
 * RiskShield Dashboard App
 *
 * This is a minimal verification component that:
 * 1. Fetches flagged transactions from the backend on component mount
 * 2. Logs the result to the browser console (for now)
 * 3. Displays a simple heading saying to check the console
 *
 * This is intentionally bare-bones — no charts, no real dashboard yet.
 * The goal is just to verify the API pipeline works:
 * React → API client → backend → database → browser console
 *
 * Future tasks will add real UI components that consume this data.
 */
function App() {
  useEffect(() => {
    /**
     * Fetch flags on component mount.
     *
     * useEffect with an empty dependency array means this runs once,
     * when the component first loads (not on every render, not when
     * component props change). This is the standard React pattern for
     * "run once on mount" logic like fetching initial data.
     */
    const fetchFlags = async () => {
      try {
        console.log("Fetching flagged transactions...");
        const flags = await getFlags({ limit: 100 });
        console.log("✓ Successfully fetched flags:", flags);
        console.log(`Found ${flags.length} flagged transactions`);
      } catch (error) {
        console.error("✗ Failed to fetch flags:", error);
        console.error("Check the Network tab in DevTools for details");
      }
    };

    fetchFlags();
  }, []); // Empty dependency array = run once on mount

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>RiskShield Dashboard</h1>
      <p>
        This is a verification step for the API pipeline.
      </p>
      <p style={{ color: "#666", fontSize: "14px" }}>
        Open the browser DevTools (F12) and check the <strong>Console</strong> tab.
        You should see "✓ Successfully fetched flags" with an array of transaction objects.
      </p>
      <p style={{ color: "#666", fontSize: "14px" }}>
        <strong>Common error:</strong> "Access to fetch has been blocked by CORS policy"
        — this means the backend's CORS middleware isn't configured correctly. Check
        that backend/app/main.py includes the CORS middleware and the backend was restarted.
      </p>
    </div>
  );
}

export default App;
