import React from 'react';

interface ProjectProps {
  id: number;
}

function Project({ id }: ProjectProps) {
  return (
    <div className={'project-container'}>
      <h4 className={'project-title'}>Elm App #{id}</h4>
      <div className={'project-body'}>
        <p>Description of the app etc...</p>
      </div>
      <div className={'project-footer'}>
        <p>Tags ...</p>
      </div>
    </div>
  );
}

export default Project;