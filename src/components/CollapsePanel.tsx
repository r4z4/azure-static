import { useState } from "react";

export interface CollapsePanelProps {
    panelData: PanelData;
}

export interface PanelData {
    name: String;
    date: String;
    category: String;
    documents: PanelDocument[];
}

export interface PanelDocument {
    id: number;
    filename: String;
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
    <div className="movie">
      <p>{panelData.name + " " + panelData.category}</p>
      <span className="showMore" onClick={() => setExpanded(!expanded)}>
        <b>Directory Name</b>
      </span>
      {expanded ? (
        <div className="expandable">
            <p>{panelData.date}</p>
            <p>Actors:</p>
            <ul>
            {panelData.documents.map((doc: PanelDocument) => (
                <li key={doc.id}>{doc.filename}</li>
            ))}
            </ul>
        </div>
        ) : null
      } 
    </div>
  );
}

export default CollapsePanel;