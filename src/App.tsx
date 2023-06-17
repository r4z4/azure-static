import React from 'react';
import ml_2 from './ml_2.png';
import pb from './assets/p_b.png'
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={ml_2} className="App-logo" alt="logo" />
        <div style={{backgroundImage: `url(${pb})`,backgroundSize:"contain"}}>
          <p style={{marginTop: '12px'}}>
            Various Curated NLP trial runs.
          </p>
          <p>&nbsp;</p>
        </div>
        <p className="subText">For now you can just select 'Just For Now' to visit the current page (hosted on GitHub)</p>
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
