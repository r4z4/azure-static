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
    path: `/articles/Run_01`,
    element: <MdArticle title={'Run_01'} />,
  },
  {
    path: `/articles/Run_02`,
    element: <MdArticle title={'Run_02'} />,
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
    <RouterProvider router={router} />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();