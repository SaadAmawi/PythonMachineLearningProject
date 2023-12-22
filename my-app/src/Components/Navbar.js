
import './Navbar.css';





function Navbar() {

    return(

    <nav className="navbar">
    
    {/* <div className="nav-container"> */}
        
        <h1 className="logo">
            <i class="fa-solid fa-arrow-trend-up"></i> 
            &nbsp;Stock-z</h1>

        <ul className="nav-menu">
        <li className="nav-item">
        <button to="/Home.js" className="nav-links"> 
        Home
        </button>
        </li>
        <li className="nav-item">
        <button to="/Home.js" className="nav-links"> 
        Watchlist
        </button>
        </li>
        <li className="nav-item">
        <button to="/Home.js" className="nav-links"> 
        PricePredictor
        </button>
        </li>
        </ul>

     


    {/* </div> */}
    

    </nav>

    );
} export default Navbar