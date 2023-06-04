import React from 'react'
import Run01 from '../assets/article_images/run_01.png'

export interface SidePanelImageDisplayProps {
    imagePaths: string[];
}

// Create map eventually
const getImage = (path: string) => {
  if (path === '../assets/folder_closed.svg') {
    return Run01
  }
  if (path === '../assets/folder_open.svg') {
    return Run01
  }
  if (path === '../assets/article_images/run_01.png') {
    return Run01
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