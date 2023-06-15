import React from 'react'
import ReactMarkdown from 'react-markdown'
import SidePanelImageDisplay from './SidePanelImageDisplay'
import style from './markdown-styles.module.css';
import dfHtml from '../utils/Dataframes'

interface MdArticleProps {
    title: string;
    subDir: string;
}

function MdArticle({ title, subDir }: MdArticleProps) {
    const mdPath = require(`../articles/${subDir}/${title}.md`)
    const [terms, setTerms] = React.useState('')
    const [html, setHtml] = React.useState('')

    const [expanded, setExpanded] = React.useState(false);

    const imagesPath = `../assets/article_images/${subDir}/${title}/`

    React.useEffect(() => {
        if (subDir === 'glove') { 
          if (title === 'run_03') {
            setHtml(dfHtml['run3html']) 
          }
          if (title === 'run_04') {
            setHtml(dfHtml['run4html']) 
          }
        }
        if (subDir === 'topic-modeling') { 
          if (title === '01_transformers') {
            setHtml(dfHtml['tm_1_html_1'] + "<br />" + dfHtml['tm_1_html_2'] + "<br />" + dfHtml['tm_1_html_3'] + "<br />" + dfHtml['tm_1_html_4']) 
          }
        }
        if (subDir === 'trec') { 
          if (title === 'trec_eda') {
            setHtml(dfHtml['trec_eda_1'] + "<br />" + dfHtml['trec_eda_2'] + "<br />" + dfHtml['trec_eda_3'] + "<br />" + dfHtml['trec_eda_4'] + "<br />" + dfHtml['trec_eda_5']) 
          }
        }
        if (subDir === 'trivia') { 
          if (title === 'lda_trivia') {
            setHtml(dfHtml['lda_trivia_1'] + "<br />" + dfHtml['lda_trivia_2'] + "<br />" + dfHtml['lda_trivia_3'] + "<br />" + dfHtml['lda_trivia_4']) 
          }
        }
        fetch(mdPath).then((response) => response.text()).then((text) => {
            
            setTerms(text)
        })
        // setTerms(markdownTable)
    }, [mdPath, subDir, title])

    return (
      <div className='grid-container'>
        <div>
          <button className="togglebtn" onClick={() => setExpanded(!expanded)}>☰</button>
          <ReactMarkdown className={style.reactMarkDown} children={terms} />
        </div>
        {expanded ? (
          <div className='flex-container'>
            <SidePanelImageDisplay html={html} imagesPath={imagesPath} />
          </div>  
          ) : null
        }
      </div>
    )
  }

export default MdArticle