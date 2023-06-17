import '../App.css';
import CollapsePanel, {PanelData, PanelDocument} from './CollapsePanel'
import EmbeddingImg from '../assets/word_embedding.png'
import TMWordCloud from '../assets/tm_wordcloud.png'

const trecEda: PanelDocument              = {id: 0, filename: 'TREC_EDA', url: '/articles/trec/trec_eda'}
const trecAug: PanelDocument              = {id: 1, filename: 'TREC_AUG', url: '/articles/trec/trec_aug',}
const runOneDocOne: PanelDocument         = {id: 2,  filename: 'Run 01',  url: '/articles/trec/run_01',}
const runOneDocTwo: PanelDocument         = {id: 3,  filename: 'Run 02',  url: '/articles/trec/run_02',}
const runTwoDocOne: PanelDocument         = {id: 4,  filename: 'Run 01',  url: '/articles/glove/run_01',}
const runTwoDocTwo: PanelDocument         = {id: 5,  filename: 'Run 02',  url: '/articles/glove/run_02',}
const runTwoDocThree: PanelDocument       = {id: 6,  filename: 'Run 03',  url: '/articles/glove/run_03',}
const runTwoDocFour: PanelDocument        = {id: 7,  filename: 'Run 04',  url: '/articles/glove/run_04',}
const TopicModelingDocOne: PanelDocument  = {id: 8,  filename: '01_Transformers',  url: '/articles/topic-modeling/01_transformers',}
const TopicModelingDocTwo: PanelDocument  = {id: 9,  filename: '02_LDA',  url: '/articles/topic-modeling/02_lda',}
const TriviaDocOne: PanelDocument         = {id: 10,  filename: 'LDA_Trivia',  url: '/articles/trivia/lda_trivia',}
const GenerateEmbeddings: PanelDocument   = {id: 11,  filename: 'Generate',  url: '/articles/embeddings/generate',}

interface ArticleHomeProps {
  active?: boolean;
}

function ArticleHome({ active = false }: ArticleHomeProps) {

  const runOneData: PanelData = {
    name: 'TREC Dataset',
    date: '05-22-2023',
    desc: 'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.',
    bgColor: '#fee6dd',
    category: 'NLP',
    documents: [trecEda, trecAug, runOneDocOne, runOneDocTwo],
  }

  const runTwoData: PanelData = {
    name: 'GloVe',
    date: '05-29-2023',
    desc: 'Several Runs Using GloVe (2014) Word Embeddings.',
    bgColor: '#effedd',
    category: 'NLP',
    documents: [runTwoDocOne, runTwoDocTwo, runTwoDocThree, runTwoDocFour],
  }

  const TopicModelingData: PanelData = {
    name: 'Topic Modeling',
    date: '05-11-2023',
    desc: 'Topic Modeling Techniques: Transformers, LDA, SKMeans',
    bgColor: '#fbddfe',
    img: TMWordCloud,
    category: 'NLP',
    documents: [TopicModelingDocOne, TopicModelingDocTwo],
  }

  const TriviaData: PanelData = {
    name: 'Trivia Dataset',
    date: '04-17-2023',
    desc: 'Trivia Question Dataset: Classification, Linear Discriminant Analysis',
    bgColor: '#fefddd',
    category: 'NLP',
    documents: [TriviaDocOne],
  }

  const WordEmbeddings: PanelData = {
    name: 'Word Embeddings',
    date: '04-10-2023',
    desc: 'All Things Embeddings.',
    bgColor: '#ddfefc',
    img: EmbeddingImg,
    category: 'NLP',
    documents: [GenerateEmbeddings],
  }

  return (
    <div className="App">
        <div className={'article-grid-container'}>
          <CollapsePanel panelData={runOneData}></CollapsePanel>
          <CollapsePanel panelData={runTwoData}></CollapsePanel>
          <CollapsePanel panelData={TopicModelingData}></CollapsePanel>
          <CollapsePanel panelData={TriviaData}></CollapsePanel>
          <CollapsePanel panelData={WordEmbeddings}></CollapsePanel>
        </div>
    </div>
  );
}

export default ArticleHome;