import React from 'react';
import articles from '../articles.json'

interface ArticleProps {
  id: number;
}

interface TypedArticles {
    [index: string]: TypedArticle;
}

interface TypedArticle {
    title: string
    author: string
    date: string
    body: string
}

const typedArticles: TypedArticles = articles

function Article({ id }: ArticleProps) {
  return (
    <div className={'article-container'}>
      <h4 className={'article-title'}>{typedArticles[String(id)].title}</h4>
      <p>{typedArticles[String(id)].author}</p>
      <div className={'article-body'}>
        <span className={'byline'}>
            {typedArticles[String(id)].body}
        </span>
      </div>
    </div>
  );
}

export default Article;