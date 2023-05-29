import React from 'react'
import ReactMarkdown from 'react-markdown'
import style from './markdown-styles.module.css';

interface MdArticleProps {
    title: string;
}

function MdArticle({ title }: MdArticleProps) {
    const mdPath = require(`../articles/${title}.md`)
    const [terms, setTerms] = React.useState('')

    React.useEffect(() => {
        fetch(mdPath).then((response) => response.text()).then((text) => {
            
            setTerms(text)
        })
        // setTerms(markdownTable)
    }, [])

    return (
      <div className="content">
        <ReactMarkdown className={style.reactMarkDown} children={terms} />
      </div>
    )
  }

export default MdArticle