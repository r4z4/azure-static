import React from 'react'
import ReactMarkdown from 'react-markdown'
import SidePanelImageDisplay from './SidePanelImageDisplay'
import style from './markdown-styles.module.css';

interface MdArticleProps {
    title: string;
    subDir: string;
}

const imagePaths = ['../assets/folder_closed.svg', '../assets/folder_open.svg', '../assets/article_images/run_01.png']

function MdArticle({ title, subDir }: MdArticleProps) {
    const mdPath = require(`../articles/${subDir}/${title}.md`)
    const [terms, setTerms] = React.useState('')

    React.useEffect(() => {
        fetch(mdPath).then((response) => response.text()).then((text) => {
            
            setTerms(text)
        })
        // setTerms(markdownTable)
    }, [mdPath])

    return (
      <div className="grid-container">
        <div>
          <ReactMarkdown className={style.reactMarkDown} children={terms} />
        </div>
        <div className='flex-container'>
          <SidePanelImageDisplay imagePaths={imagePaths} />
        </div>
      </div>
    )
  }

export default MdArticle