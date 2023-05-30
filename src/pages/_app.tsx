import Head from 'next/head'
import '../styles/App.css'
import { AppProps } from 'next/app';

interface PageProps {
    // Props
 }

export default function App({ Component, pageProps }: AppProps<PageProps>) {
    return (
        <div>
            <Head>
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <meta name="theme-color" content="#000000" />
                <meta name="description" content="Web site created using create-react-app" />
                <meta charSet="utf-8" />
                <title>r4z4 dot xyz</title>
            </Head>
            <Component {...pageProps} />
        </div>
    )
}