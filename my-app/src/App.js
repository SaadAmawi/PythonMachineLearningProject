import React from 'react';
import Navbar from './Components/Navbar';
import './App.css';
import Home from './Components/Pages/Home';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';


import { useLocation } from 'react-router-dom';

function App() {
  const location = useLocation();
  return (
   
      <>
        <Navbar />
        <Routes location={location} key={location.pathname}>
          <Route path='/' exact element={<Home/>} key={Router.pathName}/>
        </Routes>
        </>
  );
}

export default App;