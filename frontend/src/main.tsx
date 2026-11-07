import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";
import { applyTheme, loadTheme } from "./theme";

applyTheme(loadTheme());      // before the first paint: a dark page never flashes light

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
