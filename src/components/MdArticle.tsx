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

const tm_1_html_1 = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border=1 class=dataframe><thead><tr style=text-align:right><th><th>Topic<th>Doc<tbody><tr><th>0<td>-1<td>Name the mascot of Austin College ? What was t...<tr><th>1<td>0<td>Name the F1 racer with relative as Ralf Schuma...<tr><th>2<td>1<td>Which country's largest city is Lima? Which st...<tr><th>3<td>2<td>How many races have the horses bred by Jacques...<tr><th>4<td>3<td>Give me all tv shows which are based in boston...</table></div>"
const tm_1_html_2 = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border=1 class=dataframe><thead><tr style=text-align:right><th><th>Doc<th>Topic<th>Doc_ID<tbody><tr><th>0<td>How many movies did Stanley Kubrick direct?<td>19<td>0<tr><th>1<td>Which city's foundeer is John Forbes?<td>44<td>1<tr><th>2<td>What is the river whose mouth is in deadsea?<td>38<td>2<tr><th>3<td>What is the allegiance of John Kotelawala ?<td>44<td>3<tr><th>4<td>How many races have the horses bred by Jacques...<td>2<td>4</table></div>"
const tm_1_html_3 = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border=1 class=dataframe><thead><tr style=text-align:right><th><th>Topic<th>Size<tbody><tr><th>0<td>-1<td>1338<tr><th>45<td>44<td>203<tr><th>18<td>17<td>203<tr><th>22<td>21<td>194<tr><th>30<td>29<td>137<tr><th>20<td>19<td>102<tr><th>39<td>38<td>96<tr><th>12<td>11<td>95<tr><th>40<td>39<td>86<tr><th>10<td>9<td>75</table></div>"
const tm_1_html_4 = 
  "<div><style scoped>.dataframe tbody tr th:only-of-type{vertical-align:middle}.dataframe tbody tr th{vertical-align:top}.dataframe thead th{text-align:right}</style><table border=1 class=dataframe><thead><tr style=text-align:right><th><th>Topic<th>Size<tbody><tr><th>0<td>-1<td>18506<tr><th>47<td>46<td>1289<tr><th>52<td>51<td>624<tr><th>51<td>50<td>593<tr><th>48<td>47<td>361<tr><th>17<td>16<td>312<tr><th>1<td>0<td>232<tr><th>12<td>11<td>223<tr><th>50<td>49<td>221<tr><th>28<td>27<td>216</table></div>"

function MdArticle({ title, subDir }: MdArticleProps) {
    const mdPath = require(`../articles/${subDir}/${title}.md`)
    const [terms, setTerms] = React.useState('')
    const [html, setHtml] = React.useState('')

    const [expanded, setExpanded] = React.useState(false);

    const imagesPath = `../assets/article_images/${subDir}/${title}/`

    React.useEffect(() => {
        if (subDir === 'glove') { 
          if (title === 'run_03') {
            setHtml(run3html) 
          }
          if (title === 'run_04') {
            setHtml(run4html) 
          }
        }
        if (subDir === 'topic-modeling') { 
          if (title === '01_transformers') {
            setHtml(tm_1_html_1 + "<br />" + tm_1_html_2 + "<br />" + tm_1_html_3 + "<br />" + tm_1_html_4) 
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