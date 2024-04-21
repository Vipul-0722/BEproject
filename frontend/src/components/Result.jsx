import React, { useEffect, useState } from "react";
import Loader from "./Loader";
import { useSpeechSynthesis } from "react-speech-kit";
import TransButton from "./TransButton";
import Upload from "./Upload";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import axios from "axios";
// import "../topbar.css";

const Result = (props) => {
  const [preview, setPreview] = useState();
  const [caption, setCaption] = useState(); // Changing caption on UI
  const [cap, setCap] = useState(); // Constant caption
  const { speak } = useSpeechSynthesis();
  const [bool1, setBool] = useState(false);

  const callback = (lang) => {
    setCaption(lang);
  };
  const fetchCaption = async () => {
    console.log(props.img, "props");
    const formData = new FormData();
    formData.append("file", props.img);
    console.log(formData);
    try {
      const url = `http://localhost:5000/after`;
      const response = await axios.post(url, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log(response.data);
      // const data = await response.json();

      setCaption(response.data.description);
    } catch (err) {
      console.log(err);
    }
  };
  useEffect(() => {
    setPreview(URL.createObjectURL(props.img));

    fetchCaption();
  }, []);

  const handleClick = () => {
    setBool(true);
  };

  return (
    <>
      {!bool1 && (
        <div className="result-page">
          {/* <video id="bg-video" src="../bgvid.mp4" autoplay loop muted></video> */}
          <div className="result-window" style={{ position: "reative" }}>
            <button
              style={{ color: "black", marginLeft: "-31rem" }}
              className="result-logout"
              onClick={handleClick}
            >
              Go back
            </button>
            <h1 className="result-heading">Result page</h1>
            {preview && (
              <img className="result-image" src={preview} alt="image" />
            )}
            {caption ? <p className="result-caption">{caption}</p> : <Loader />}
          </div>
        </div>
      )}

      {bool1 && <Upload />}
    </>
  );
};

export default Result;
