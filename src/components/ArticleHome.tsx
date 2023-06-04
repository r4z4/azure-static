import '../App.css';
import MdArticlePrev from './MdArticlePrev';
import CollapsePanel, {PanelData, PanelDocument} from './CollapsePanel'

const runOneDocOne: PanelDocument = {
  id: 1,
  filename: 'Run 01',
  url: '/articles/trec/Run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.'} concepts={['Multilabel Confusion Matrix', 'Tokenizer/CountVectorizer']} hangups={['Concat/Merge/Join']} />
}

const runOneDocTwo: PanelDocument = {
  id: 2,
  filename: 'Run 02',
  url: '/articles/trec/Run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for (TREC) Question Classification dataset.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocOne: PanelDocument = {
  id: 3,
  filename: 'Run 01',
  url: '/articles/glove/Run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['GloVe', 'Scaler']} hangups={['Vocab Size']} />
}

const runTwoDocTwo: PanelDocument = {
  id: 4,
  filename: 'Run 02',
  url: '/articles/glove/Run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'Warm Embedding']} hangups={['Wordnet']} />
}

const runTwoDocThree: PanelDocument = {
  id: 5,
  filename: 'Run 03',
  url: '/articles/glove/Run_03',
  previewComponent: <MdArticlePrev title={'Run_03'} desc={'Third Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocFour: PanelDocument = {
  id: 6,
  filename: 'Run 04',
  url: '/articles/glove/Run_04',
  previewComponent: <MdArticlePrev title={'Run_04'} desc={'Fouth Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

interface ArticleHomeProps {
  active?: boolean;
}

function ArticleHome({ active = false }: ArticleHomeProps) {

  const runOneData: PanelData = {
    name: 'TREC Dataset',
    date: '05-22-2023',
    desc: 'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.',
    category: 'NLP',
    documents: [runOneDocOne, runOneDocTwo],
  }

  const runTwoData: PanelData = {
    name: 'GloVe',
    date: '05-29-2023',
    desc: 'Several Runs Using GloVe (2014) Word Embeddings.',
    category: 'NLP',
    documents: [runTwoDocOne, runTwoDocTwo, runTwoDocThree, runTwoDocFour],
  }

  return (
    <div className="App">
        <nav>
          <a href="/">Home</a>
          {' '}
          <a href="/projects">Projects</a>
        </nav>
        {/* <div className={'article-grid-container'}>
          <Article id={1} />
          <Article id={2} />
          <Article id={2} />
        </div> */}
        <div className={'article-grid-container'}>
          <CollapsePanel panelData={runOneData}></CollapsePanel>
          <CollapsePanel panelData={runTwoData}></CollapsePanel>
        </div>
    </div>
  );
}

export default ArticleHome;