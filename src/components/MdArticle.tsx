import React from 'react'
import ReactMarkdown from 'react-markdown'
import SidePanelImageDisplay from './SidePanelImageDisplay'
import style from './markdown-styles.module.css';

interface MdArticleProps {
    title: string;
    subDir: string;
}
const run3html = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border='1' class='dataframe'><thead><tr style='text-align:right'><th></th><th>newsgroup</th><th>subject</th></tr></thead><tbody><tr><th>0</th><td>autos</td><td>saturn's pricing policy</td></tr><tr><th>10</th><td>autos</td><td>are bmw's worth the price</td></tr><tr><th>12</th><td>autos</td><td>re headlights problem</td></tr><tr><th>14</th><td>autos</td><td>left turn signal won't stop automaticaly</td></tr><tr><th>16</th><td>autos</td><td>what is volvo</td></tr></tbody></table></div>"
const run4html = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border='1' class='dataframe'><thead><tr style='text-align:right'><th></th><th>newsgroup</th><th>subject</th></tr></thead><tbody><tr><th>0</th><td>autos</td><td>saturn's pricing policy</td></tr><tr><th>10</th><td>autos</td><td>are bmw's worth the price</td></tr><tr><th>12</th><td>autos</td><td>re headlights problem</td></tr><tr><th>14</th><td>autos</td><td>left turn signal won't stop automaticaly</td></tr><tr><th>16</th><td>autos</td><td>what is volvo</td></tr></tbody></table></div>"

function MdArticle({ title, subDir }: MdArticleProps) {
    const mdPath = require(`../articles/${subDir}/${title}.md`)
    const [terms, setTerms] = React.useState('')
    const [html, setHtml] = React.useState('')

    const imagePaths = [`../assets/article_images/${subDir}/${title}.png`]

    React.useEffect(() => {
        if (subDir === 'glove') { 
          if (title === 'run_03') {
            setHtml(run3html) 
          }
          if (title === 'run_04') {
            setHtml(run4html) 
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
          <ReactMarkdown className={style.reactMarkDown} children={terms} />
        </div>
        <div className='flex-container'>
          <SidePanelImageDisplay html={html} imagePaths={imagePaths} />
        </div>
      </div>
    )
  }

export default MdArticle