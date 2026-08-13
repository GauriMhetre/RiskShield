import React, { useState, useCallback } from 'react';
import { getFlags } from '../api/client';
import { usePolling } from '../hooks/usePolling';

export default function FlaggedTransactionFeed({ selectedFlag, onSelectFlag }) {
  const [flags, setFlags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchFlags = useCallback(async () => {
    try {
      const data = await getFlags({ limit: 20 });
      setFlags(data);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch flagged transactions");
    } finally {
      // UX Choice: We only set loading to true initially in the state.
      // We explicitly do NOT set loading to true at the start of this fetch function.
      // If we did, the entire table would disappear and be replaced by a "Loading..."
      // message every 7 seconds, which would be a jarring and terrible experience.
      // Instead, we let the existing data stay on screen while the background poll happens.
      setLoading(false);
    }
  }, []);

  usePolling(fetchFlags, 7000);

  if (loading) {
    return <div>Loading flagged transactions...</div>;
  }

  if (error) {
    return <div style={{ color: 'red', padding: '10px', border: '1px solid red' }}>Error: {error}</div>;
  }

  if (flags.length === 0) {
    return <div>No flagged transactions yet</div>;
  }

  return (
    <div style={{ marginTop: '20px' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #ddd' }}>
        <thead>
          <tr style={{ backgroundColor: '#f9f9f9' }}>
            <th style={thStyle}>Time</th>
            <th style={thStyle}>User ID</th>
            <th style={thStyle}>Amount</th>
            <th style={thStyle}>Country</th>
            <th style={thStyle}>Risk Score</th>
            <th style={thStyle}>Model Version</th>
          </tr>
        </thead>
        <tbody>
          {flags.map((flag) => {
            const isSelected = selectedFlag && selectedFlag.txn_id === flag.txn_id;
            return (
              <tr 
                key={flag.txn_id} 
                onClick={() => onSelectFlag(flag)}
                style={{ 
                  cursor: 'pointer',
                  backgroundColor: isSelected ? '#e6f7ff' : 'transparent',
                  transition: 'background-color 0.2s'
                }}
              >
                <td style={tdStyle}>{new Date(flag.scored_at).toLocaleString()}</td>
                <td style={tdStyle} title={flag.user_id}>
                  {flag.user_id.length > 8 ? `${flag.user_id.substring(0, 8)}...` : flag.user_id}
                </td>
                <td style={tdStyle}>${parseFloat(flag.amount).toFixed(2)}</td>
                <td style={tdStyle}>{flag.country}</td>
                <td style={tdStyle}>{parseFloat(flag.risk_score).toFixed(2)}</td>
                <td style={tdStyle}>{flag.model_version}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = {
  padding: '12px',
  textAlign: 'left',
  borderBottom: '2px solid #ddd'
};

const tdStyle = {
  padding: '10px',
  borderBottom: '1px solid #ddd'
};
