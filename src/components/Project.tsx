import React from 'react';

interface ProjectProps {
  active?: boolean;
}

function Project({ active = false }: ProjectProps) {
  return (
    <div className={'project-container'}>
      <h4 className={'project-title'}>Elm App #1</h4>
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