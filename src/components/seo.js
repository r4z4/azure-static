import Head from 'next/head'

export default function SEO( {description, title, siteTitle }) {
        return (
            <Head>
                <title>{ `${title} | ${siteTitle}`}</title>
                <meta name="description" content={description} />
                <meta property="twitter:title" content={title} />
                <meta property="twitter:description" content={description} />
            </Head>
        )
}