import React from 'react'
import Trec01 from '../assets/article_images/trec/run_01.png'
import Trec02 from '../assets/article_images/trec/run_01.png'
import Glove01 from '../assets/article_images/glove/run_01.png'
import Glove02 from '../assets/article_images/glove/run_01.png'
import Glove03 from '../assets/article_images/glove/run_01.png'
import Glove04 from '../assets/article_images/glove/run_01.png'

export interface SidePanelImageDisplayProps {
    imagePaths: string[];
}

// Create map eventually
const getImage = (path: string) => {
  if (path === '../assets/article_images/glove/run_01.png') {
    return Glove01
  }
  if (path === '../assets/article_images/glove/run_02.png') {
    return Glove02
  }
  if (path === '../assets/article_images/glove/run_03.png') {
    return Glove03
  }
  if (path === '../assets/article_images/glove/run_04.png') {
    return Glove04
  }
  if (path === '../assets/article_images/trec/run_01.png') {
    return Trec01
  }
  if (path === '../assets/article_images/trec/run_02.png') {
    return Trec02
  }
}

function SidePanelImageDisplay({ imagePaths }: SidePanelImageDisplayProps) {

  return (
    <aside className='side-panel-aside'>
      <div className='side-panel-container'>
        <ul className='side-panel-list'>
          <h3>Artifacts</h3>
          {imagePaths.map((path: string) => (
            <div className='mapped-image'>
              <img className={'side-panel-image'} src={getImage(path)} alt={path} />
            </div>
          ))}
        </ul>
      </div>
    </aside>
  );
}

export default SidePanelImageDisplay;