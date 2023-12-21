// import React,{useState, useEffect} from 'react'
// import { Link } from 'react-router-dom'
// import { Button } from './Button';
import './Navbar.css';
// import './Button.css';




function Navbar() {

    return(

    <nav className="navbar">
    
    {/* <div className="nav-container"> */}
        
        <h1 className="logo">
            <i class="fa-solid fa-arrow-trend-up"></i> 
            &nbsp;Stockz</h1>

        <ul className="nav-menu">
        <li className="nav-item">
        <a to="/" className="nav-links"> 
        Home
        </a>
        </li>
        <li className="nav-item">
        <a to="/" className="nav-links"> 
        Watchlist
        </a>
        </li>
        <li className="nav-item">
        <a to="/" className="nav-links"> 
        PricePredictor
        </a>
        </li>
        </ul>

     


    {/* </div> */}
    

    </nav>

    );
} export default Navbar