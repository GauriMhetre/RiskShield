import React from 'react';

export default function TransactionDetailPanel({ transaction }) {
  if (!transaction) {
    return (
      <div style={panelStyle}>
        <h3>Transaction Details</h3>
        <p style={{ color: '#666' }}>Select a transaction from the feed to view its details.</p>
      </div>
    );
  }

  // Calculate top reason
  const getTopReason = (features) => {
    if (!features) return "Not available";
    
    const candidates = ['amount_zscore', 'device_mismatch', 'country_mismatch', 'geo_distance_km'];
    let maxVal = -1;
    let topReason = "None";
    
    for (const key of candidates) {
      if (features[key] !== undefined && features[key] !== null) {
        const absVal = Math.abs(features[key]);
        if (absVal > maxVal) {
          maxVal = absVal;
          topReason = key;
        }
      }
    }
    
    return topReason !== "None" ? `${topReason} (${maxVal.toFixed(2)})` : "None";
  };

  const topReason = getTopReason(transaction.feature_snapshot);

  return (
    <div style={panelStyle}>
      <h3>Transaction Details</h3>
      
      <div style={{ marginBottom: '15px' }}>
        <strong>ID:</strong> {transaction.txn_id} <br/>
        <strong>Risk Score:</strong> {parseFloat(transaction.risk_score).toFixed(2)} <br/>
        <strong style={{ color: '#d32f2f' }}>Top Reason:</strong> {topReason}
      </div>

      <h4>Feature Snapshot</h4>
      {transaction.feature_snapshot ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #eee' }}>
          <thead>
            <tr style={{ backgroundColor: '#fafafa' }}>
              <th style={thStyle}>Feature</th>
              <th style={thStyle}>Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(transaction.feature_snapshot).map(([key, val]) => (
              <tr key={key}>
                <td style={tdStyle}>{key}</td>
                <td style={tdStyle}>{typeof val === 'number' ? val.toFixed(4) : String(val)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ color: '#666', fontStyle: 'italic' }}>Feature snapshot not available.</p>
      )}
    </div>
  );
}

const panelStyle = {
  border: '1px solid #ccc',
  borderRadius: '4px',
  padding: '20px',
  marginTop: '20px',
  backgroundColor: '#fff',
  boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
};

const thStyle = {
  padding: '8px',
  textAlign: 'left',
  borderBottom: '1px solid #eee',
  fontSize: '0.9em'
};

const tdStyle = {
  padding: '8px',
  borderBottom: '1px solid #eee',
  fontSize: '0.9em'
};
