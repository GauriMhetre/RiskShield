import React from 'react';
import FlaggedTransactionFeed from './components/FlaggedTransactionFeed';

function App() {
  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>RiskShield Dashboard</h1>
      <FlaggedTransactionFeed />
    </div>
  );
}

export default App;
