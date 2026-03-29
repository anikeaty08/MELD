import React from "react";
import ReactDOM from "react-dom/client";
import Prism from "prismjs";
import "prismjs/themes/prism.css";
import "prismjs/components/prism-bash";
import "prismjs/components/prism-python";
import "prismjs/components/prism-json";
import "./styles.css";
import App from "./App.jsx";

window.Prism = Prism;

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
