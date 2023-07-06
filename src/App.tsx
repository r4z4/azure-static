import React from 'react';
import ml_2 from './ml_2.png';
import './App.css';
import './HomeAnimation.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
      <div className='animation-example'>
        <div className='item'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item -type2'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item -type2'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item -type2'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='item -type2'>
            <div className='line'></div>
            <div className='dot'></div>
            <div className='circle'></div>
        </div>
        <div className='center'>
            <div className='circle'></div>
            <div className='circle img_c'><img src={ml_2} className="App-logo" alt="logo" /></div>
            <div className='circle'></div>
        </div>
      </div>
          <p style={{marginTop: '12px', textAlign: 'center'}}>
            <span className="text-glow text-gradient-elm">Various Curated NLP trial runs.</span><br />
            <span className="text-glow text-gradient-elixir">Ventures Into Mojo (AI First)</span>
          </p>
        <p className="subText">For now just links to site hosted by GitHub</p>
        <div className={'links-container'}>
        <a
          className="route-link text-glow text-gradient"
          href="/articles"
          rel="noopener noreferrer"
        >
          Articles
        </a>
        <a
          className="route-link text-glow text-gradient-fire"
          href="https://r4z4.github.io"
          target="_blank"
          rel="noopener noreferrer"
        >
          Projects
        </a>
        <a
          className="route-link text-glow text-gradient-emerald"
          href="/about"
          rel="noopener noreferrer"
        >
          About
        </a>
        </div>
      </header>
    </div>
  );
}

export default App;
