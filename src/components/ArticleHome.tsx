import '../App.css';
import MdArticlePrev from './MdArticlePrev';
import CollapsePanel, {PanelData, PanelDocument} from './CollapsePanel'

const panelDocOne: PanelDocument = {
  id: 1,
  filename: 'Run 01',
  url: '/articles/Run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.'} concepts={['Multilabel Confusion Matrix', 'Tokenizer/CountVectorizer']} hangups={['Concat/Merge/Join']} />
}

const panelDocTwo: PanelDocument = {
  id: 2,
  filename: 'Run 02',
  url: '/articles/Run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for (TREC) Question Classification dataset.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
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
    documents: [panelDocOne, panelDocTwo],
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
        </div>
    </div>
  );
}

export default ArticleHome;