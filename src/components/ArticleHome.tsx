import '../App.css';
import Article from './Article';

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
        <div className={'article-grid-container'}>
          <Article />
          <Article />
          <Article />
        </div>
    </div>
  );
}

export default ArticleHome;