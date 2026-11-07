import { afterEach, describe, expect, it, vi } from "vitest";
import { applyTheme, loadTheme, resolveTheme } from "./theme";

afterEach(() => {
  window.localStorage.clear();
  delete document.documentElement.dataset.theme;
  vi.restoreAllMocks();
});

describe("theme", () => {
  it("follows the system until the reader chooses", () => {
    expect(loadTheme()).toBe("system");
    window.localStorage.setItem("study-assistant-theme", "dark");
    expect(loadTheme()).toBe("dark");
    window.localStorage.setItem("study-assistant-theme", "purple");
    expect(loadTheme()).toBe("system");
  });

  it("resolves system to the system's mode", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
  });

  it("sets data-theme and color-scheme on <html>", () => {
    applyTheme("dark", false);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
    applyTheme("system", false);
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("still works when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(loadTheme()).toBe("system");
  });
});
