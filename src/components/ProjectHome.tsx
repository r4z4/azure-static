import '../App.css';
import Project from './Project'
import CollapsePanel, {PanelData, PanelDocument} from './CollapsePanel'

interface ProjectHomeProps {
  active?: boolean;
}

const panelDocOne: PanelDocument = {
  id: 1,
  filename: 'Run 01',
  url: '/articles/Run_01'
}

const panelDocTwo: PanelDocument = {
  id: 2,
  filename: 'Run 02',
  url: '/articles/Run_02'
}

function ProjectHome({ active = false }: ProjectHomeProps) {

  const testPanelData: PanelData = {
    name: 'Panel Name',
    date: '05-22-2020',
    desc: 'E.g. Panel Desc',
    category: 'Government',
    documents: [panelDocOne, panelDocTwo]
  }
  
  return (
    <div className='home-page'>
        <nav>
          <a href='/'>Home</a>
          {' '}
          <a href="/articles">Articles</a>
        </nav>
        <div className={'article-grid-container'}>
          <Project id={1} />
          <Project id={2} />
          <Project id={3}  />
        </div>
        <div className={'article-grid-container'}>
          <CollapsePanel panelData={testPanelData}></CollapsePanel>
        </div>
    </div>
  );
}

export default ProjectHome;