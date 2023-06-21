import React from 'react';
import ml_2 from './ml_2.png';
import pb from './assets/p_b.png'
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
          <p style={{marginTop: '12px'}}>
            Various Curated NLP trial runs.
          </p>
        <p className="subText">Projects - For now you can just select 'Just For Now' to visit the current page (hosted on GitHub)</p>
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
          href="https://r4z4.github.io"
          target="_blank"
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
