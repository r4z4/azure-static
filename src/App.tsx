import React from 'react';
import gear from './gear.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={gear} className="App-logo" alt="logo" />
        <p>
          Not Sure What We're Doing Here Yet
        </p>
        <a
          className="gh-link"
          href="https://r4z4.github.io"
          target="_blank"
          rel="noopener noreferrer"
        >
          Just For Now
        </a>
      </header>
    </div>
  );
}

export default App;
