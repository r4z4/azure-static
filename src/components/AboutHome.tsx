import '../App.css';
import CVPanel, {CVPanelData, CVPanelDocument} from './CVPanel'

interface AboutHomeProps {
  active?: boolean;
}

const AboutMe: CVPanelData = {
  name: 'About Me',
  date: '07/2023',
  desc: "Not Sure How I'm Gonna Do This Yet",
  bgColor: '#801a00',
  category: 'About',
  documents: [],
}

function AboutHome({ active = false }: AboutHomeProps) {

  return (
    <div className="App">
        <div className={'cv-container'}>
          <CVPanel panelData={AboutMe}></CVPanel>
        </div>
    </div>
  );
}

export default AboutHome;