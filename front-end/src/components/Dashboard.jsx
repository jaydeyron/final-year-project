// src/components/Dashboard.jsx
import React from 'react';

function Dashboard({ selectedIndex }) {
  return (
    <main>
      <h2>{selectedIndex ? `${selectedIndex} Data` : 'Select an Index'}</h2>
      <div className="chart">
        {selectedIndex ? `Displaying chart for ${selectedIndex}` : 'No data available'}
      </div>
    </main>
  );
}

export default Dashboard;
