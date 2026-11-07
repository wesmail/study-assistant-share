import { useEffect, useState } from "react";

/** Light, dark, or whatever the system says (Windows' "Choose your mode"). The choice is kept
 *  in this browser only; the colours themselves are in theme.css. */
export type Theme = "light" | "dark" | "system";

export const THEMES: Theme[] = ["light", "dark", "system"];
const KEY = "study-assistant-theme";
const QUERY = "(prefers-color-scheme: dark)";

export function loadTheme(): Theme {
  try {
    const saved = window.localStorage.getItem(KEY);
    return THEMES.includes(saved as Theme) ? (saved as Theme) : "system";
  } catch {
    return "system";        // storage blocked: follow the system
  }
}

function saveTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(KEY, theme);
  } catch {
    // not kept across reloads, but still applied now
  }
}

function systemIsDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(QUERY).matches;
}

export function resolveTheme(theme: Theme, systemDark: boolean = systemIsDark()): "light" | "dark" {
  return theme === "system" ? (systemDark ? "dark" : "light") : theme;
}

/** Sets <html data-theme> and color-scheme (scrollbars, form controls). Also called before
 *  the first render (main.tsx), so a dark page never flashes light. */
export function applyTheme(theme: Theme, systemDark: boolean = systemIsDark()): void {
  const resolved = resolveTheme(theme, systemDark);
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
}

export function useTheme(): [Theme, (theme: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(loadTheme);
  useEffect(() => {
    applyTheme(theme);
    if (theme !== "system" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(QUERY);
    const follow = () => applyTheme("system", media.matches);
    media.addEventListener("change", follow);
    return () => media.removeEventListener("change", follow);
  }, [theme]);
  const choose = (next: Theme) => {
    saveTheme(next);
    setTheme(next);
  };
  return [theme, choose];
}
