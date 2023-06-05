import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import ArticleHome from './components/ArticleHome';
import ProjectHome from './components/ProjectHome';
import MdArticle from './components/MdArticle';
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import reportWebVitals from './reportWebVitals';

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />
  },
  {
    path: "/articles",
    element: <ArticleHome />,
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
    </nav>
    <RouterProvider router={router} />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
