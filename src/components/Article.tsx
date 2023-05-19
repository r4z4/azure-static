import React from 'react';

interface ArticleProps {
  active?: boolean;
}

function Article({ active = false }: ArticleProps) {
  return (
    <div className={'article-container'}>
      <h4 className={'article-title'}>Title</h4>
      <div className={'article-body'}>
        <span className={'byline'}>
          Content
        </span>
      </div>
    </div>
  );
}

export default Article;