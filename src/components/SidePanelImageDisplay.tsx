import React from 'react'
import Trec01 from '../assets/article_images/trec/run_01/run_01.png'
import Glove01 from '../assets/article_images/glove/run_01.png'
import Glove02 from '../assets/article_images/glove/run_01.png'
import Glove03 from '../assets/article_images/glove/run_01.png'
import Glove04 from '../assets/article_images/glove/run_01.png'
import TM01 from '../assets/article_images/topic-modeling/01_transformers.png'
import TM02 from '../assets/article_images/topic-modeling/02_LDA.png'
import TREC_EDA_0 from '../assets/article_images/trec/eda/trec_eda_0.png'
import TREC_EDA_1 from '../assets/article_images/trec/eda/trec_eda_1.png'
import TREC_EDA_2 from '../assets/article_images/trec/eda/trec_eda_2.png'
import TREC_EDA_3 from '../assets/article_images/trec/eda/trec_eda_3.png'
import TREC_EDA_4 from '../assets/article_images/trec/eda/trec_eda_4.png'
import TREC_EDA_5 from '../assets/article_images/trec/eda/trec_eda_5.png'
import TREC_EDA_6 from '../assets/article_images/trec/eda/trec_eda_6.png'
import LDA_TRIVIA_0 from '../assets/article_images/trivia/lda_trivia_0.png'
import LDA_TRIVIA_1 from '../assets/article_images/trivia/lda_trivia_1.png'
import parse from 'html-react-parser';

type StringMap = { 
  [id: string]: string; 
}

const getImage: StringMap = {
  'glove/run_01.png': Glove01,
  'glove/run_02.png': Glove02,
  'glove/run_03.png': Glove03,
  'glove/run_04.png': Glove04,
  'trec/run_01.png': Trec01,
  'trec/run_02.png': Trec01,
  'topic-modeling/01_transformers.png': TM01,
  'topic-modeling/02_LDA.png': TM02,
  'trec_eda_0.png': TREC_EDA_0,
  'trec_eda_1.png': TREC_EDA_1,
  'trec_eda_2.png': TREC_EDA_2,
  'trec_eda_3.png': TREC_EDA_3,
  'trec_eda_4.png': TREC_EDA_4,
  'trec_eda_5.png': TREC_EDA_5,
  'trec_eda_6.png': TREC_EDA_6,
  'trivia/lda_trivia_0.png': LDA_TRIVIA_0,
  'trivia/lda_trivia_1.png': LDA_TRIVIA_1
};

export interface SidePanelImageDisplayProps {
    imagesPath: string;
    html: string;
}


function SidePanelImageDisplay({ imagesPath, html }: SidePanelImageDisplayProps) {

  const [modalOpen, setModalOpen] = React.useState('')

  function getFilenames(imagesPath: string) {
    switch(imagesPath) {
      case '../assets/article_images/trec/trec_eda/':
        return ['trec_eda_0.png','trec_eda_1.png','trec_eda_2.png','trec_eda_3.png','trec_eda_4.png','trec_eda_5.png', 'trec_eda_6.png'];

      case '../assets/article_images/trec/run_01/':
        return ['trec/run_01.png'];
      case '../assets/article_images/trec/run_02/':
        return ['trec/run_01.png'];

      case '../assets/article_images/glove/run_01/':
        return ['glove/run_01.png'];
      case '../assets/article_images/glove/run_02/':
        return ['glove/run_02.png'];
      case '../assets/article_images/glove/run_03/':
        return ['glove/run_03.png'];
      case '../assets/article_images/glove/run_04/':
        return ['glove/run_04.png'];

      case '../assets/article_images/trivia/lda_trivia/':
        return ['trivia/lda_trivia_0.png', 'trivia/lda_trivia_1.png'];

      case '../assets/article_images/topic-modeling/01_transformers/':
        return ['topic-modeling/01_transformers.png'];
      case '../assets/article_images/topic-modeling/02_LDA/':
        return ['topic-modeling/02_LDA.png'];

      default:
        return [''];
    }
  }
  const imageFilenames = getFilenames(imagesPath)

  console.log("gFN=", getImage['trec/run_01.png'])

  return (
    <>
      <aside aria-label="sidePanel" className='side-panel-aside'>
        <div className='side-panel-container'>
        <h3>Artifacts</h3>
          <div>
            {parse(html)}
          </div>
          
          <ul className='side-panel-list'>
            {imageFilenames.map((filename: string, index: number) => (
              <li key={index}>
                <div className='mapped-image'>
                {/* Wrap LDA image in anchor tag */}
                {getImage[filename] === TM02 ? <a href="/articles/topic-modeling/02_LDA/pyLDAvis"><img aria-label={filename} className={'side-panel-image'} src={getImage[filename]} alt={filename} /></a> : 
                  <div className="tooltip">
                    <span aria-label="tooltipText" className="tooltipText">Click to Enlarge</span>
                    <img onClick={() => setModalOpen(modalOpen === '' ? filename : '')} aria-label='side-panel-image' className={'side-panel-image'} src={getImage[filename]} alt={filename} />
                  </div>
                }
                </div>
              </li>
            ))}
          </ul>
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
            src={getImage[modalOpen]}
            alt="enlargedImg"
          />
        </dialog>
      )}
    </>
  );
}

export default SidePanelImageDisplay;