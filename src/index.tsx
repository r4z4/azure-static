import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import ArticleHome from './components/ArticleHome';
import AboutHome from './components/AboutHome';
import ProjectHome from './components/ProjectHome';
import MdArticle from './components/MdArticle';
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router-dom";
import reportWebVitals from './reportWebVitals';


const router = createBrowserRouter([
  {
    path: "/",
    element: <App />
  },
  {
    path: "/about",
    element: <AboutHome />,
  },
  {
    path: "/articles",
    element: <ArticleHome />,
  },
  {
    path: `/articles/trec/trec_eda`,
    element: <MdArticle subDir={'trec'} title={'trec_eda'} />,
  },
  {
    path: `/articles/trec/trec_aug`,
    element: <MdArticle subDir={'trec'} title={'trec_aug'} />,
  },
  {
    path: `/articles/trec/run_01`,
    element: <MdArticle subDir={'trec'} title={'run_01'} />,
  },
  {
    path: `/articles/trec/run_02`,
    element: <MdArticle subDir={'trec'} title={'run_02'} />,
  },
  {
    path: `/articles/glove/run_01`,
    element: <MdArticle subDir={'glove'} title={'run_01'} />,
  },
  {
    path: `/articles/glove/run_02`,
    element: <MdArticle subDir={'glove'} title={'run_02'} />,
  },
  {
    path: `/articles/glove/run_03`,
    element: <MdArticle subDir={'glove'} title={'run_03'} />,
  },
  {
    path: `/articles/glove/run_04`,
    element: <MdArticle subDir={'glove'} title={'run_04'} />,
  },
  {
    path: `/articles/topic-modeling/01_transformers`,
    element: <MdArticle subDir={'topic-modeling'} title={'01_transformers'} />,
  },
  {
    path: `/articles/topic-modeling/02_LDA`,
    element: <MdArticle subDir={'topic-modeling'} title={'02_LDA'} />,
  },
  {
    path: `/articles/topic-modeling/02_LDA/pyLDAvis`,
    element: <Navigate to={"/ldavis_prepared_10.html"} />,
  },
  {
    path: `/articles/trivia/lda_trivia`,
    element: <MdArticle subDir={'trivia'} title={'lda_trivia'} />,
  },
  {
    path: `/articles/embeddings/generate`,
    element: <MdArticle subDir={'embeddings'} title={'generate'} />,
  },
  {
    path: '/articles/dimred/viz',
    element: <MdArticle subDir={'dimred'} title={'viz'} />,
  },
  {
    path: `/articles/dimred/viz/pca`,
    element: <Navigate to={"/glove_pca.html"} />,
  },
  {
    path: `/articles/dimred/viz/tsne`,
    element: <Navigate to={"/glove_tsne.html"} />,
  },
  {
    path: `/articles/news/eda`,
    element: <MdArticle subDir={'news'} title={'eda'} />,
  },
  {
    path: `/articles/news/clean_run_01`,
    element: <MdArticle subDir={'news'} title={'clean_run_01'} />,
  },
  {
    path: `/articles/news/clean_run_02`,
    element: <MdArticle subDir={'news'} title={'clean_run_02'} />,
  },
  {
    path: `/articles/news/clean_run_03`,
    element: <MdArticle subDir={'news'} title={'clean_run_03'} />,
  },
  {
    path: `/articles/news/clean_run_04`,
    element: <MdArticle subDir={'news'} title={'clean_run_04'} />,
  },
  {
    path: `/articles/news/body_clean_run_01`,
    element: <MdArticle subDir={'news'} title={'body_clean_run_01'} />,
  },
  {
    path: `/articles/news/body_clean_run_02`,
    element: <MdArticle subDir={'news'} title={'body_clean_run_02'} />,
  },
  {
    path: `/articles/news/body_clean_run_03`,
    element: <MdArticle subDir={'news'} title={'body_clean_run_03'} />,
  },
  {
    path: `/articles/news/body_clean_run_04`,
    element: <MdArticle subDir={'news'} title={'body_clean_run_04'} />,
  },
  {
    path: `/articles/mojo/intro`,
    element: <MdArticle subDir={'mojo'} title={'intro'} />,
  },
  {
    path: `/articles/mojo/generate`,
    element: <MdArticle subDir={'mojo'} title={'generate'} />,
  },
  {
    path: `/articles/elixir/benchmark`,
    element: <MdArticle subDir={'elixir'} title={'benchmark'} />,
  },
  {
    path: `/articles/elixir/zip_with`,
    element: <MdArticle subDir={'elixir'} title={'zipWith'} />,
  },
  {
    path: `/articles/elixir/Concuerror`,
    element: <MdArticle subDir={'elixir'} title={'Concuerror'} />,
  },




  {
    path: "/projects",
    element: <ProjectHome />,
  }
]);

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <nav>
      <a href="/">Home</a>
      {' '}
      <a href="/projects">Projects</a>
      {' '}
      <a href="/articles">Articles</a>
      {' '}
      <a href="/about">About</a>
    </nav>
    <RouterProvider router={router} />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
