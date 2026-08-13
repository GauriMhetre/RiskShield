import React, { useState } from 'react';
import FlaggedTransactionFeed from './components/FlaggedTransactionFeed';
import RiskScoreDistribution from './components/RiskScoreDistribution';
import FlagsOverTimeChart from './components/FlagsOverTimeChart';
import TransactionDetailPanel from './components/TransactionDetailPanel';

function App() {
  const [selectedFlag, setSelectedFlag] = useState(null);

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>RiskShield Dashboard</h1>
      
      <div style={{ display: 'flex', gap: '20px', flexDirection: 'row', flexWrap: 'wrap' }}>
        <div style={{ flex: '2 1 600px' }}>
          <FlaggedTransactionFeed 
            selectedFlag={selectedFlag} 
            onSelectFlag={setSelectedFlag} 
          />
        </div>
        <div style={{ flex: '1 1 300px' }}>
          <TransactionDetailPanel transaction={selectedFlag} />
        </div>
      </div>

      <RiskScoreDistribution />
      <FlagsOverTimeChart />
    </div>
  );
}

export default App;
