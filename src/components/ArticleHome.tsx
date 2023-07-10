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
  id: 1,
  filename: 'TREC_AUG',
  url: '/articles/trec/trec_aug',
  previewComponent: <MdArticlePrev title={'TREC_AUG'} desc={'Easy Data Augmentation Techniques'} concepts={['Augmentation']} hangups={['']} />
}

const runOneDocOne: PanelDocument = {
  id: 2,
  filename: 'Run 01',
  url: '/articles/trec/run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.'} concepts={['Multilabel Confusion Matrix', 'Tokenizer/CountVectorizer']} hangups={['Concat/Merge/Join']} />
}

const runOneDocTwo: PanelDocument = {
  id: 3,
  filename: 'Run 02',
  url: '/articles/trec/run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for (TREC) Question Classification dataset.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocOne: PanelDocument = {
  id: 4,
  filename: 'Run 01',
  url: '/articles/glove/run_01',
  previewComponent: <MdArticlePrev title={'Run_01'} desc={'Initial Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['GloVe', 'Scaler']} hangups={['Vocab Size']} />
}

const runTwoDocTwo: PanelDocument = {
  id: 5,
  filename: 'Run 02',
  url: '/articles/glove/run_02',
  previewComponent: <MdArticlePrev title={'Run_02'} desc={'Second Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'Warm Embedding']} hangups={['Wordnet']} />
}

const runTwoDocThree: PanelDocument = {
  id: 6,
  filename: 'Run 03',
  url: '/articles/glove/run_03',
  previewComponent: <MdArticlePrev title={'Run_03'} desc={'Third Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const runTwoDocFour: PanelDocument = {
  id: 7,
  filename: 'Run 04',
  url: '/articles/glove/run_04',
  previewComponent: <MdArticlePrev title={'Run_04'} desc={'Fouth Run for 20_Newsgroups dataset using GloVe Embeddings.'} concepts={['Data Augmentation', 'NLTK']} hangups={['Wordnet']} />
}

const TopicModelingDocOne: PanelDocument = {
  id: 8,
  filename: '01_Transformers',
  url: '/articles/topic-modeling/01_transformers',
  previewComponent: <MdArticlePrev title={'01_Transformers'} desc={'Topic Modeling on Trivia Dataset for Surface Trivia App. Using SentenceTransformers.'} concepts={['Transformers', 'Attention', 'BERT']} hangups={['Input Shape']} />
}

const TopicModelingDocTwo: PanelDocument = {
  id: 9,
  filename: '02_LDA',
  url: '/articles/topic-modeling/02_lda',
  previewComponent: <MdArticlePrev title={'02_LDA'} desc={'Topic Modeling on Trivia Dataset for Surface Trivia App. Using LDA and visualizing with PyLDAvis.'} concepts={['Data Visualization', 'LDA']} hangups={['LDA']} />
}

const TriviaDocOne: PanelDocument = {
  id: 10,
  filename: 'LDA_Trivia',
  url: '/articles/trivia/lda_trivia',
  previewComponent: <MdArticlePrev title={'LDA_Trivia'} desc={'Calssification Using Linear Disciminant Analysis - Compare to PCA'} concepts={['LDA', 'PCA']} hangups={['Matplotlib']} />
}

const GenerateEmbeddings: PanelDocument = {
  id: 11,
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

const NewsgroupEDA: PanelDocument = {
  id: 13,
  filename: 'News_EDA',
  url: '/articles/news/eda',
  previewComponent: <MdArticlePrev title={'NewsgroupEDA'} desc={'Preprocessing and EDA for the 20_Newsgroup Dataset'} concepts={['Vocab Size, Stemming']} hangups={['Unstructured Text']} />
}

const CleanRun01: PanelDocument = {
  id: 14,
  filename: 'News_01',
  url: '/articles/news/clean_run_01',
  previewComponent: <MdArticlePrev title={'Run01'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['Meaning Retention']} />
}

const CleanRun02: PanelDocument = {
  id: 15,
  filename: 'News_02',
  url: '/articles/news/clean_run_02',
  previewComponent: <MdArticlePrev title={'Run02'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['eaning Retention']} />
}

const CleanRun03: PanelDocument = {
  id: 16,
  filename: 'News_03',
  url: '/articles/news/clean_run_03',
  previewComponent: <MdArticlePrev title={'Run03'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['eaning Retention']} />
}

const CleanRun04: PanelDocument = {
  id: 17,
  filename: 'News_04',
  url: '/articles/news/clean_run_04',
  previewComponent: <MdArticlePrev title={'Run04'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['eaning Retention']} />
}

const BodyCleanRun01: PanelDocument = {
  id: 18,
  filename: 'NewsB_01',
  url: '/articles/news/body_clean_run_01',
  previewComponent: <MdArticlePrev title={'BodyRun01'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['Meaning Retention']} />
}

const BodyCleanRun02: PanelDocument = {
  id: 19,
  filename: 'NewsB_02',
  url: '/articles/news/body_clean_run_02',
  previewComponent: <MdArticlePrev title={'BodyRun02'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['eaning Retention']} />
}

const BodyCleanRun03: PanelDocument = {
  id: 20,
  filename: 'NewsB_03',
  url: '/articles/news/body_clean_run_03',
  previewComponent: <MdArticlePrev title={'BodyRun03'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['eaning Retention']} />
}

const BodyCleanRun04: PanelDocument = {
  id: 21,
  filename: 'NewsB_04',
  url: '/articles/news/body_clean_run_04',
  previewComponent: <MdArticlePrev title={'BodyRun04'} desc={'Original Runs with Full Body data for 20 Newsgroup Dataset'} concepts={['Preprocessing']} hangups={['Leaning Retention']} />
}

const MojoIntro: PanelDocument = {
  id: 22,
  filename: 'Mojo_Intro.🔥',
  url: '/articles/mojo/intro',
  previewComponent: <MdArticlePrev title={'MojoIntro'} desc={'Intro to the Mojo Programming Language. AI First.'} concepts={['Lvalue, Rvalue, LLVM, Clang, AutoDiff']} hangups={['A Whole Bunch of C++ Concepts']} />
}

const MojoGen: PanelDocument = {
  id: 23,
  filename: 'Mojo_Gen.🔥',
  url: '/articles/mojo/generate',
  previewComponent: <MdArticlePrev title={'MojoGen'} desc={'A Generate Embedding Run Using the Mojo Syntax. Way More Difficult Than I Thought :)'} concepts={['Lvalue, Rvalue, LLVM, Clang, AutoDiff']} hangups={['A Whole Bunch of C++ Concepts']} />
}

const Benchmarking: PanelDocument = {
  id: 24,
  filename: 'Benchmark',
  url: '/articles/elixir/benchmark',
  previewComponent: <MdArticlePrev title={'Benchmark'} desc={'Benchmarking our Function Calls using Benchee'} concepts={['Benchmark, Profile']} hangups={['.exs Script']} />
}

const zipWith: PanelDocument = {
  id: 25,
  filename: 'zipWith',
  url: '/articles/elixir/zip_with',
  previewComponent: <MdArticlePrev title={'zipWith'} desc={'Utilizing the Enum.zip_with Fucntion'} concepts={['Enum, Zip']} hangups={['Haskell']} />
}

interface ArticleHomeProps {
  active?: boolean;
}

function ArticleHome({ active = false }: ArticleHomeProps) {

  const runOneData: PanelData = {
    name: 'TREC Dataset',
    date: '05-22-2023',
    desc: 'Initial Run for the Text REtrieval Conference (TREC) Question Classification dataset.',
    bgColor: '#b3b300',
    category: 'NLP',
    documents: [trecEda, trecAug, runOneDocOne, runOneDocTwo],
  }

  const runTwoData: PanelData = {
    name: 'GloVe',
    date: '05-29-2023',
    desc: 'Several Runs Using GloVe (2014) Word Embeddings.',
    bgColor: '#002699',
    category: 'NLP',
    documents: [runTwoDocOne, runTwoDocTwo, runTwoDocThree, runTwoDocFour],
  }

  const TopicModelingData: PanelData = {
    name: 'Topic Modeling',
    date: '05-11-2023',
    desc: 'Topic Modeling Techniques: Transformers, LDA, SKMeans',
    bgColor: '#801a00',
    category: 'NLP',
    documents: [TopicModelingDocOne, TopicModelingDocTwo],
  }

  const TriviaData: PanelData = {
    name: 'Trivia Dataset',
    date: '04-17-2023',
    desc: 'Trivia Question Dataset: Classification, Linear Discriminant Analysis',
    bgColor: '#4d3900',
    category: 'NLP',
    documents: [TriviaDocOne],
  }

  const WordEmbeddings: PanelData = {
    name: 'Word Embeddings',
    date: '04-10-2023',
    desc: 'All Things Embeddings.',
    bgColor: '#001a09',
    category: 'NLP',
    documents: [GenerateEmbeddings],
  }

  const DimReduction: PanelData = {
    name: 'Dimensionality Reduction',
    date: '06-10-2023',
    desc: 'Brief Exploration of the various techniques, and why they are needed.',
    bgColor: '#33001a',
    category: 'NLP',
    documents: [EmbeddingViz],
  }

  const Newsgroup: PanelData = {
    name: 'The 20_Newsgroup Data',
    date: '06-10-2023',
    desc: 'This is the dataset that we use in the initial GloVe runs. These notebooks are some of the original runs that were done with the dataset and the preprocessing steps as well,',
    bgColor: '#4d2600',
    category: 'NLP',
    documents: [NewsgroupEDA, CleanRun01, CleanRun02, CleanRun03, CleanRun04, BodyCleanRun01, BodyCleanRun02, BodyCleanRun03, BodyCleanRun04],
  }

  const MojoRuns: PanelData = {
    name: 'Mojo 🔥 Runs',
    date: '07-04-2023',
    desc: 'Mojo is a superset of Python. It is the programming language for the full stack AI architecture from Modular. This one will take some time.',
    bgColor: '#001a09',
    category: 'NLP',
    documents: [MojoIntro, MojoGen],
  }

  const ElixirArticles: PanelData = {
    name: 'Elixir Articles',
    date: '07-08-2023',
    desc: 'Using Benchee Library to Benchmark function calls. Particularly interested here in the peformance of using Lists versus Tuples.',
    bgColor: '#001a09',
    category: 'Elixir',
    documents: [Benchmarking, zipWith],
  }

  return (
    <div className="App">
        <div className={'article-grid-container'}>
          <CollapsePanel panelData={MojoRuns}></CollapsePanel>
          <CollapsePanel panelData={Newsgroup}></CollapsePanel>
          <CollapsePanel panelData={runOneData}></CollapsePanel>
          <CollapsePanel panelData={runTwoData}></CollapsePanel>
          <CollapsePanel panelData={TopicModelingData}></CollapsePanel>
          <CollapsePanel panelData={TriviaData}></CollapsePanel>
          <CollapsePanel panelData={WordEmbeddings}></CollapsePanel>
          <CollapsePanel panelData={DimReduction}></CollapsePanel>
          <CollapsePanel panelData={ElixirArticles}></CollapsePanel>
        </div>
    </div>
  );
}

export default ArticleHome;