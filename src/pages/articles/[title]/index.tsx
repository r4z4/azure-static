import React from 'react'
import ReactMarkdown from 'react-markdown'
import { useRouter } from 'next/router'
import style from '../../../styles/markdown-styles.module.css';

interface MdArticleProps {
    title: string;
}

function MdArticle() {
    const { query } = useRouter();
    // const mdPath = require(`../../articles/${query['title']}.md`)
    const [terms, setTerms] = React.useState('')

    // React.useEffect(() => {
    //     fetch(mdPath).then((response) => response.text()).then((text) => {
            
    //         setTerms(text)
    //     })
    //     // setTerms(markdownTable)
    // }, [mdPath])

    return (
      <div className="content">
        {/* <ReactMarkdown className={style.reactMarkDown} children={terms} /> */}
        <p>Oof</p>
      </div>
    )
  }

export default MdArticle