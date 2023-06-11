import React from 'react'
import Trec01 from '../assets/article_images/trec/run_01.png'
import Trec02 from '../assets/article_images/trec/run_01.png'
import Glove01 from '../assets/article_images/glove/run_01.png'
import Glove02 from '../assets/article_images/glove/run_01.png'
import Glove03 from '../assets/article_images/glove/run_01.png'
import Glove04 from '../assets/article_images/glove/run_01.png'
import TM01 from '../assets/article_images/topic-modeling/01_transformers.png'
import TM02 from '../assets/article_images/topic-modeling/02_LDA.png'
import parse from 'html-react-parser';

export interface SidePanelImageDisplayProps {
    imagePaths: string[];
    html: string;
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
  if (path === '../assets/article_images/topic-modeling/01_transformers.png') {
    return TM01
  }
  if (path === '../assets/article_images/topic-modeling/02_LDA.png') {
    return TM02
  }
}

function SidePanelImageDisplay({ imagePaths, html }: SidePanelImageDisplayProps) {

  const [modalOpen, setModalOpen] = React.useState('')

  return (
    <>
      <aside className='side-panel-aside'>
        <div className='side-panel-container'>
          <ul className='side-panel-list'>
            <h3>Artifacts</h3>
            {imagePaths.map((path: string) => (
              <div className='mapped-image'>
              {/* Wrap LDA image in anchor tag */}
              {getImage(path) === TM02 ? <a href="/articles/topic-modeling/02_LDA/pyLDAvis"><img className={'side-panel-image'} src={getImage(path)} alt={path} /></a> : 
                <div className="tooltip">
                  <span className="tooltipText">Click to Enlarge</span>
                  <img onClick={() => setModalOpen(modalOpen === '' ? path : '')} className={'side-panel-image'} src={getImage(path)} alt={path} />
                </div>
              }
              </div>
            ))}
          </ul>

          <div>
            {parse(html)}
          </div>
        </div>
      </aside>

      {modalOpen && modalOpen !== '' && (
        <dialog
          className="dialog"
          style={{ position: 'absolute' }}
          open
          onClick={() => setModalOpen('')}
        >
          <img
            className="image"
            src={getImage(modalOpen)}
            alt="enlargedImg"
          />
        </dialog>
      )}
    </>
  );
}

export default SidePanelImageDisplay;