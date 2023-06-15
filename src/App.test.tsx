import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event'
import MdArticlePrev from './components/MdArticlePrev';
import SidePanelImageDisplay from './components/SidePanelImageDisplay';
import App from './App';

test('renders Projects link', () => {
  render(<App />);
  const linkElement = screen.getByText(/Projects/i);
  expect(linkElement).toBeInTheDocument();
});

test('renders Articles link', () => {
  render(<App />);
  const linkElement = screen.getByText(/Articles/i);
  expect(linkElement).toBeInTheDocument();
});

test('renders Markdown Article', () => {
  render(<MdArticlePrev title={'02_LDA'} desc={'Topic Modeling on Trivia Dataset for Surface Trivia App. Using LDA and visualizing with PyLDAvis.'} concepts={['Data Visualization', 'LDA']} hangups={['LDA']} />);
  const span = screen.getByRole('heading', {name: /02_LDA/i})
  expect(span).toBeInTheDocument();
});

{/* name property refers to 'accessable name of element' - the aria-label (and some others too) */}
const testImgsPath = 'assets/article_images/trec/run_01/'

test('renders SidePanel img', () => {
  render(<SidePanelImageDisplay html={''} imagesPath={testImgsPath} />);
  const heading = screen.getByRole('heading', {name: /Artifacts/i})
  const comp = screen.getByRole('complementary', {name: /sidePanel/i})
  const img = screen.getByRole('img', {name: "assets/article_images/trec/run_02.png"})
  userEvent.hover(img)
  expect(screen.getByText(/Click to Enlarge/i)).toBeInTheDocument()
  expect(heading).toBeInTheDocument();
  expect(comp).toBeInTheDocument();
  expect(img).toBeInTheDocument();
});