// src/components/Sidebar.jsx
import React from 'react';

function Sidebar({ indices, onSelectIndex }) {
  return (
    <aside>
      <h2>Indices</h2>
      <ul>
        {indices.map((index) => (
          <li key={index} onClick={() => onSelectIndex(index)}>
            {index}
          </li>
        ))}
      </ul>
    </aside>
  );
}

export default Sidebar;
