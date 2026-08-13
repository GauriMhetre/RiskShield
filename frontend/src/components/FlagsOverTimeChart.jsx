import React, { useState, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { getFlags } from '../api/client';
import { usePolling } from '../hooks/usePolling';

export function bucketFlagsByHour(flags) {
  const countsByHour = {};
  
  flags.forEach(flag => {
    // Parse to Date, truncate minutes, seconds, ms
    const date = new Date(flag.scored_at);
    date.setMinutes(0, 0, 0);
    
    // hourKey for sorting (toISOString is sortable)
    const hourKey = date.toISOString();
    
    if (!countsByHour[hourKey]) {
      // hourLabel for display (e.g. "Aug 8, 2 PM")
      const hourLabel = date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        hour12: true
      });
      
      countsByHour[hourKey] = {
        hourKey,
        hourLabel,
        count: 0
      };
    }
    countsByHour[hourKey].count += 1;
  });
  
  // This task's bucketing differs from bucketRiskScores()'s always-show-all-buckets approach.
  // The score range is fixed (0-1), but the time range is open-ended and unknown in advance.
  // Showing every empty hour since the beginning of time wouldn't make sense, so we only
  // include hours that actually have at least one flag.
  
  // Convert map values to array and sort chronologically by hourKey.
  // This sort step is required because object/map iteration order isn't guaranteed
  // to match chronological order, and Recharts will draw connecting lines in whatever
  // order the array is in.
  const bucketedArray = Object.values(countsByHour).sort((a, b) => {
    if (a.hourKey < b.hourKey) return -1;
    if (a.hourKey > b.hourKey) return 1;
    return 0;
  });
  
  return bucketedArray;
}

export default function FlagsOverTimeChart() {
  const [flags, setFlags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchFlags = useCallback(async () => {
    try {
      const data = await getFlags({ limit: 200 });
      setFlags(data);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch flagged transactions");
    } finally {
      setLoading(false);
    }
  }, []);

  usePolling(fetchFlags, 7000);

  const bucketedData = useMemo(() => bucketFlagsByHour(flags), [flags]);

  if (loading) {
    return <div>Loading flags over time...</div>;
  }

  if (error) {
    return <div style={{ color: 'red', padding: '10px', border: '1px solid red' }}>Error: {error}</div>;
  }

  if (flags.length === 0) {
    return <div>No flagged transactions yet</div>;
  }

  return (
    <div style={{ marginTop: '40px' }}>
      <h2>Flags Over Time</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={bucketedData}>
          <XAxis dataKey="hourLabel" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Line 
            type="monotone" 
            dataKey="count" 
            stroke="#8884d8" 
            dot={true} 
            activeDot={{ r: 8 }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
