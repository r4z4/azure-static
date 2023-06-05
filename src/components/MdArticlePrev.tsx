import React from 'react'

interface MdArticlePrevProps {
    title: string
    desc: string
    concepts: string[]
    hangups: string[]
}

function MdArticle({ title, desc, concepts, hangups }: MdArticlePrevProps) {

    // const [terms, setTerms] = React.useState('')

    // React.useEffect(() => {
    //     fetch(mdPath).then((response) => response.text()).then((text) => {
            
    //         setTerms(text)
    //     })
    //     // setTerms(markdownTable)
    // }, [])

  return (
    <div className={'article-container'}>
      <h4 className={'article-title'}>{title}</h4>
      <div className={'article-body'}>
        <span className={'byline'}>
            <p>{desc}</p>
        </span>
      <div className={'prev-list'}>
        <h6>Concepts</h6>
        <ul>
          {concepts.map((concept)=><li>{concept}</li>)}
        </ul>
      </div>
      <div className={'prev-list'}>
        <h6>HangUps</h6>
        <ul>
          {hangups.map((hangup)=><li>{hangup}</li>)}
        </ul>
      </div>
      </div>
    </div>
    )
  }

export default MdArticle