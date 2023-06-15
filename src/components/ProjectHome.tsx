import '../App.css';
import Project from './Project'

interface ProjectHomeProps {
  active?: boolean;
}

function ProjectHome({ active = false }: ProjectHomeProps) {
  
  return (
    <div className='home-page'>
        <div className={'article-grid-container'}>
          <Project id={1} />
          <Project id={2} />
          <Project id={3}  />
        </div>
    </div>
  );
}

export default ProjectHome;