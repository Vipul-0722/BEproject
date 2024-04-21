import React from "react";
import styles from "../styles.module.css";
import projectLogoVideo from "../videos/abc.mp4";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className={styles.topContainer}>
      <video id="bg-video" autoPlay loop muted>
        <source src={projectLogoVideo} type="video/mp4" />
      </video>

      <div className={styles.info}>
        <h1>Caption Magic</h1>
        <h2>The Next-Gen Image Insight Generation</h2>
        <div className={styles.description}>
          <h3>
            Allow our AI to handle the conversation! Create descriptions for
            images quickly and easily in just a few seconds.
          </h3>
          <h3>
            Please feel free to upload your images and see how CaptionMagic
            creates captions for them.
          </h3>
        </div>
        <div className={styles.getStarted}>
          <Link className={styles.btn} to="/upload">
            Get Started!
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
