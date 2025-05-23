import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import SpeechRecorder from './components/SpeechRecorder';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SpeechRecorder />
    </ThemeProvider>
  );
};

export default App; 