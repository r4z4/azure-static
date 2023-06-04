import { useState } from "react";
import FolderClosedIcon from '../assets/folder_closed.svg'
import FolderOpenIcon from '../assets/folder_open.svg'

export interface CollapsePanelProps {
    panelData: PanelData;
}

export interface PanelData {
    name: String;
    date: String;
    desc: String;
    category: String;
    documents: PanelDocument[];
}

export interface PanelDocument {
    id: number;
    filename: String;
    url: string;
    previewComponent?: JSX.Element;
}

// const panelData {
//     name = 'Panel Name',
//     date = '05-22-2020',
//     category = 'Government',
//     documents = ['Doc1', 'Doc2', 'Doc3']
// }

function CollapsePanel({ panelData }: CollapsePanelProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="collapse-panel">
      <p>{panelData.name + " " + panelData.category}</p>
      <span className="show-more" onClick={() => setExpanded(!expanded)}>
        {expanded ? <img width={'35px'} src={FolderOpenIcon} alt='folderClosedIcon'/> : <img width={'35px'} src={FolderClosedIcon} alt='folderOpenIcon'/>}
        <b className='panel-dir-name'>{panelData.name}</b>
      </span>
        <p>Last Updated: {panelData.date}</p>
      {expanded ? (
        <div className="expandable">
            <p>{panelData.desc}</p>
            <ul className='panel-doc-list'>
            {panelData.documents.map((doc: PanelDocument) => (
              <>
              <div className='file-grid'>
                <li className="prev-toggle" key={doc.id}><a href={doc.url}>{doc.filename}</a></li>
                <div className="hide" key={doc.previewComponent?.key}>{doc.previewComponent}</div>
              </div>
              </>
            ))}
            </ul>
        </div>
        ) : null
      } 
    </div>
  );
}

export default CollapsePanel;