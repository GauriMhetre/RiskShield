import React, { useState, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { getFlags } from '../api/client';
import { usePolling } from '../hooks/usePolling';

/**
 * Pure function to bucket risk scores into 10 fixed bands (0.0-0.1, 0.1-0.2, etc.)
 * Intentionally pure (input array in, new array out, no side effects) so its
 * correctness can be reasoned about independently of the chart rendering.
 */
export function bucketRiskScores(flags) {
  // Initialize all 10 buckets with count 0 so the chart x-axis is always consistent
  const buckets = [
    { bucket: "0.0-0.1", count: 0 },
    { bucket: "0.1-0.2", count: 0 },
    { bucket: "0.2-0.3", count: 0 },
    { bucket: "0.3-0.4", count: 0 },
    { bucket: "0.4-0.5", count: 0 },
    { bucket: "0.5-0.6", count: 0 },
    { bucket: "0.6-0.7", count: 0 },
    { bucket: "0.7-0.8", count: 0 },
    { bucket: "0.8-0.9", count: 0 },
    { bucket: "0.9-1.0", count: 0 }
  ];

  flags.forEach(flag => {
    // Parse to float just in case it's a string in the API response
    const score = parseFloat(flag.risk_score);
    if (isNaN(score)) return;

    // Calculate bucket index (0 to 9)
    let index = Math.floor(score * 10);
    
    // Edge case: exactly 1.0 goes into the last bucket (index 9)
    if (index >= 10) {
      index = 9;
    }
    // Also clamp below 0 just in case
    if (index < 0) {
      index = 0;
    }

    buckets[index].count += 1;
  });

  return buckets;
}

export default function RiskScoreDistribution() {
  const [flags, setFlags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchFlags = useCallback(async () => {
    try {
      const data = await getFlags({ limit: 200 }); // Larger limit for a broader sample
      setFlags(data);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch flagged transactions");
    } finally {
      // Loading only true on first fetch
      setLoading(false);
    }
  }, []);

  usePolling(fetchFlags, 7000);

  if (loading) {
    return <div>Loading risk score distribution...</div>;
  }

  if (error) {
    return <div style={{ color: 'red', padding: '10px', border: '1px solid red' }}>Error: {error}</div>;
  }

  if (flags.length === 0) {
    return <div>No flagged transactions yet</div>;
  }

  const bucketedData = bucketRiskScores(flags);

  return (
    <div style={{ marginTop: '40px' }}>
      <h2>Risk Score Distribution</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={bucketedData}>
          <XAxis dataKey="bucket" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="#3182ce" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
