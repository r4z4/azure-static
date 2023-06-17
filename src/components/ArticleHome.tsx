import '../App.css';
import MdArticlePrev from './MdArticlePrev';
import CollapsePanel, {PanelData, PanelDocument} from './CollapsePanel'

const trecEda: PanelDocument = {
  id: 0,
  filename: 'TREC_EDA',
  url: '/articles/trec/trec_eda',
  previewComponent: <MdArticlePrev title={'TREC_EDA'} desc={'Exploratory Data Analysis for the TREC Dataset.'} concepts={['EDA', 'WordCloud']} hangups={['Document Term Matrix & Grouping']} />
}

const trecAug: PanelDocument = {
  id: 0,
  filename: 'TREC_AUG',
  url: '/articles/trec/trec_aug',
  previewComponent: <MdArticlePrev title={'TREC_AUG'} desc={'Easy Data Augmentation Techniques'} concepts={['Augmentation']} hangups={['']} />
}

const runOneDocOne: PanelDocument = {
  id: 1,
  filename: 'Run 01',
  url: '/articles/trec/run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.'} concepts={['Multilabel Confusion Matrix', 'Tokenizer/CountVectorizer']} hangups={['Concat/Merge/Join']} />
}

const runOneDocTwo: PanelDocument = {
  id: 2,
  filename: 'Run 02',
  url: '/articles/trec/run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for (TREC) Question Classification dataset.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocOne: PanelDocument = {
  id: 3,
  filename: 'Run 01',
  url: '/articles/glove/run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['GloVe', 'Scaler']} hangups={['Vocab Size']} />
}

const runTwoDocTwo: PanelDocument = {
  id: 4,
  filename: 'Run 02',
  url: '/articles/glove/run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'Warm Embedding']} hangups={['Wordnet']} />
}

const runTwoDocThree: PanelDocument = {
  id: 5,
  filename: 'Run 03',
  url: '/articles/glove/run_03',
  previewComponent: <MdArticlePrev title={'Run_03'} desc={'Third Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocFour: PanelDocument = {
  id: 6,
  filename: 'Run 04',
  url: '/articles/glove/run_04',
  previewComponent: <MdArticlePrev title={'Run_04'} desc={'Fouth Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const TopicModelingDocOne: PanelDocument = {
  id: 7,
  filename: '01_Transformers',
  url: '/articles/topic-modeling/01_transformers',
  previewComponent: <MdArticlePrev title={'01_Transformers'} desc={'Topic Modeling on Trivia Dataset for Surface Trivia App. Using SentenceTransformers.'} concepts={['Transformers', 'Attention', 'BERT']} hangups={['Input Shape']} />
}

const TopicModelingDocTwo: PanelDocument = {
  id: 8,
  filename: '02_LDA',
  url: '/articles/topic-modeling/02_lda',
  previewComponent: <MdArticlePrev title={'02_LDA'} desc={'Topic Modeling on Trivia Dataset for Surface Trivia App. Using LDA and visualizing with PyLDAvis.'} concepts={['Data Visualization', 'LDA']} hangups={['LDA']} />
}

const TriviaDocOne: PanelDocument = {
  id: 9,
  filename: 'LDA_Trivia',
  url: '/articles/trivia/lda_trivia',
  previewComponent: <MdArticlePrev title={'LDA_Trivia'} desc={'Calssification Using Linear Disciminant Analysis - Compare to PCA'} concepts={['LDA', 'PCA']} hangups={['Matplotlib']} />
}

const GenerateEmbeddings: PanelDocument = {
  id: 10,
  filename: 'Generate',
  url: '/articles/embeddings/generate',
  previewComponent: <MdArticlePrev title={'Generate'} desc={'Use a NN to Genreate Custom Word Embeddings'} concepts={['Embeddings']} hangups={['Vocab Size']} />
}

const EmbeddingViz: PanelDocument = {
  id: 12,
  filename: 'Embedding Visualizations',
  url: '/articles/dimred/viz',
  previewComponent: <MdArticlePrev title={'Generate'} desc={'Use a NN to Genreate Custom Word Embeddings'} concepts={['Embeddings']} hangups={['Vocab Size']} />
}

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
    category: 'NLP',
    documents: [GenerateEmbeddings],
  }

  const DimReduction: PanelData = {
    name: 'Dimensionality Reduction',
    date: '06-10-2023',
    desc: 'Brief Exploration of the various techniques, and why they are needed.',
    bgColor: '#e0e0eb',
    category: 'NLP',
    documents: [EmbeddingViz],
  }

  return (
    <div className="App">
        <div className={'article-grid-container'}>
          <CollapsePanel panelData={runOneData}></CollapsePanel>
          <CollapsePanel panelData={runTwoData}></CollapsePanel>
          <CollapsePanel panelData={TopicModelingData}></CollapsePanel>
          <CollapsePanel panelData={TriviaData}></CollapsePanel>
          <CollapsePanel panelData={WordEmbeddings}></CollapsePanel>
          <CollapsePanel panelData={DimReduction}></CollapsePanel>
        </div>
    </div>
  );
}

export default ArticleHome;