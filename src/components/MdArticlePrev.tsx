import React from 'react'
import ReactMarkdown from 'react-markdown'

interface MdArticlePrevProps {
    title: string
    desc: string
}

function MdArticle({ title, desc }: MdArticlePrevProps) {

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
            <p><a href={`/articles/${title}`}>Link to Article</a></p>
            <p>{desc}</p>
        </span>
      </div>
    </div>
    )
  }

export default MdArticle