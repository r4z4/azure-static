import '../App.css';
import Article from './Article';
import MdArticlePrev from './MdArticlePrev';

interface ArticleHomeProps {
  active?: boolean;
}

function ArticleHome({ active = false }: ArticleHomeProps) {
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
          <MdArticlePrev title={'Run_01'} desc={'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.'} concepts={['Multilabel Confusion Matrix', 'Tokenizer/CountVectorizer']} hangups={['Concat/Merge/Join']} />
          <MdArticlePrev title={'Run_02'} desc={'Second Run for (TREC) Question Classification dataset.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
          {/* <MdArticlePrev title={'run_01'} />
          <MdArticlePrev title={'run_01'} /> */}
        </div>
    </div>
  );
}

export default ArticleHome;