import React from 'react';
import FlaggedTransactionFeed from './components/FlaggedTransactionFeed';
import RiskScoreDistribution from './components/RiskScoreDistribution';

function App() {
  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>RiskShield Dashboard</h1>
      <FlaggedTransactionFeed />
      <RiskScoreDistribution />
    </div>
  );
}

export default App;
