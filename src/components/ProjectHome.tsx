import '../App.css';
import Project from './Project'

interface ProjectHomeProps {
  active?: boolean;
}

function ProjectHome({ active = false }: ProjectHomeProps) {
  return (
    <div className="home-page">
        <nav>
          <a href="/">Home</a>
          {' '}
          <a href="/articles">Articles</a>
        </nav>
        <div className={'article-grid-container'}>
          <Project />
          <Project />
          <Project />
        </div>
    </div>
  );
}

export default ProjectHome;