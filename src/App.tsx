import React from 'react';
import gear from './gear.svg';
import './App.css';
import Article from './components/Article';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={gear} className="App-logo" alt="logo" />
        <p>
          Not Sure What We're Doing Here Yet
        </p>
        <a
          className="route-link"
          href="https://r4z4.github.io"
          target="_blank"
          rel="noopener noreferrer"
        >
          Just For Now
        </a>
        <div className={'links-container'}>
        <a
          className="route-link"
          href="/articles"
          rel="noopener noreferrer"
        >
          Articles
        </a>
        <a
          className="route-link"
          href="/projects"
          rel="noopener noreferrer"
        >
          Projects
        </a>
        </div>
      </header>
    </div>
  );
}

export default App;
