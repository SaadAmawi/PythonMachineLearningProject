import React from 'react';
import './Footer.css';
import { Link } from 'react-router-dom';

function Footer() {
  return (
    <div className='footer-container'>
      <section className='footer-subscription'>
        <p className='footer-subscription-heading'>
          Join the newsletter to receive updates and upcoming features
        </p>
        <p className='footer-subscription-text'>
          You can unsubscribe at any time.
        </p>
        <div className='input-areas'>
          <form>
            <button className='Sub'>Subscribe</button>
          </form>
        </div>
      </section>
      <div class='footer-links'>
        <div className='footer-link-wrapper'>
          <div class='footer-link-items'>
            <h2>About Us</h2>
            <Link to='/sign-up'>How it works</Link>
            <Link to=''>Testimonials</Link>
            <Link to=''>Careers</Link>
            <Link to=''>Investors</Link>
            <Link to=''>Terms of Service</Link>
          </div>
          </div>
          <div className='footer-link-wrapper'>
          <div class='footer-link-items'>
            <h2>Contact Us</h2>
            <Link to=''>Contact</Link>
            <Link to=''>Support</Link>
            <Link to=''>Destinations</Link>
            <Link to=''>Sponsorships</Link>
          </div>
        </div>
        
        <div className='footer-link-wrapper'>
          <div class='footer-link-items'>
            <h2>Videos</h2>
            <Link to=''>Submit Video</Link>
            <Link to=''>Ambassadors</Link>
            <Link to=''>Agency</Link>
            <Link to=''>Influencer</Link>
          </div>
     
        </div>
      </div>
      <section class='social-media'>
        <div class='social-media-wrap'>
          <div class='footer-logo'>
            <Link to='' className='social-logo'>
            MemorEyez 
              
            </Link>
          </div>
          <small class='website-rights'>MemorEyez © 2023</small>
          <div class='social-icons'>
            <Link
              class='social-icon-link-blue'
              to=''
              aria-label='Facebook'
            >
              <i class='fab fa-facebook-f' />
            </Link>
            <Link
              class='social-icon-link-insta'
              to=''
              aria-label='Instagram'
            >
              <i class='fab fa-instagram' />
            </Link>
            <Link
              class='social-icon-link-youtube'
              to=''
              aria-label='Youtube'
            >
              <i class='fab fa-youtube' />
            </Link>
            <Link
              class='social-icon-link-twitter'
              to=''
              aria-label='Twitter'
            >
              <i class='fab fa-twitter' />
            </Link>
            <Link
              class='social-icon-link-blue'
              to=''
              aria-label='LinkedIn'
            >
              <i class='fab fa-linkedin' />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Footer;